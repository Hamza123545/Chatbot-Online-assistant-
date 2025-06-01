import chainlit as cl
import os
from dotenv import load_dotenv
import litellm
import logging
import re
import json # For parsing tool arguments

\
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('online_store_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

litellm.telemetry = False 
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


MODEL = "openrouter/google/gemini-2.5-flash-preview-05-20"

if not API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")
if not MODEL:
    raise ValueError("Please set the MODEL environment variable.")

BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER = "openrouter"

litellm.register_model({
    MODEL: {
        "api_base": BASE_URL,
        "api_key": API_KEY,
        "provider": PROVIDER
    }
})


online_store_kb = {
    "shipping policy": "Standard shipping within Pakistan takes 3-5 business days. Express options are available. You'll receive a tracking number once your order ships.",
    "return policy": "We offer a 7-day return policy for unused items in original packaging. Please visit our website's 'Returns' page for full details and to initiate a return.",
    "refund status": "Refunds are processed within 5-7 business days after we receive and inspect the returned item. You'll be notified via email once your refund is issued.",
    "payment methods": "We accept Cash on Delivery (COD), bank transfers, and all major credit/debit cards (Visa, MasterCard) through secure online payment gateways.",
    "order tracking": "You can track your order using the tracking number sent to your email. If you haven't received it, please check your spam folder or contact us.",
    "product availability": "Our website shows real-time product availability. If an item is out of stock, you can sign up for 'back in stock' notifications on the product page.",
    "account login issue": "If you're having trouble logging in, try resetting your password using the 'Forgot Password' link. If that doesn't work, contact support.",
    "discounts and promotions": "Stay updated on our latest discounts and promotions by subscribing to our newsletter or following our social media pages!",
    "customer support": "For any specific order issues, product questions, or other concerns, please email us at **support@onlinestore.com** or call us at **0312-3456789** during business hours (Mon-Fri, 10 AM - 6 PM PKT).",
    "delivery charges": "Delivery charges vary based on your location and order value. You'll see the exact shipping cost calculated at checkout before payment."
}


def search_online_store_kb(query: str) -> str:
    """
    Searches the online store's knowledge base for information about shipping,
    returns, payment, order tracking, product availability, or customer support.
    """
    logger.info(f"Searching online store KB for: '{query}'")
    query_lower = query.lower()
    found_answers = []

    for key, value in online_store_kb.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            found_answers.append(value)
    
    if found_answers:
        return "\n\n".join(found_answers) + "\n\nIs there anything else I can assist you with regarding your shopping experience?"
    else:
        return "I couldn't find information about that. For specific order details or if you need more help, please contact our support team directly:\n\n* **Email:** `support@onlinestore.com`\n* **Phone:** `0312-3456789`"


class OnlineStoreChatbotAgent:
    def __init__(self, name: str, instructions: str, model: str, tools_list: list):
        self.name = name
        self.instructions = instructions
        self.model = model
        self._callable_tools = {
            "search_online_store_kb": search_online_store_kb
        }
        self.litellm_tools_config = tools_list
        logger.info(f"Agent '{self.name}' initialized with {len(self._callable_tools)} tools.")

    async def generate_response(self, messages: list[dict]) -> str:
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=API_KEY,
                api_base=BASE_URL,
                tools=self.litellm_tools_config,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=300 
            )

            message_response = response.choices[0].message
            content = message_response.content

            if hasattr(message_response, "tool_calls") and message_response.tool_calls:
                tool_call = message_response.tool_calls[0]
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if func_name in self._callable_tools:
                    tool_output = self._callable_tools[func_name](**args)
                    
                    messages.append(message_response)
                    messages.append({
                        "role": "tool",
                        "name": func_name,
                        "content": tool_output
                    })
                    
                    follow_up_response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        api_key=API_KEY,
                        api_base=BASE_URL,
                        tools=self.litellm_tools_config,
                        tool_choice="auto",
                        temperature=0.7,
                        max_tokens=300
                    )
                    content = follow_up_response.choices[0].message.content
                else:
                    content = "Sorry, I couldn't process that. Can you please rephrase?"
            
            return content if content else "Hmm, I'm not sure how to answer that. Can you try asking in another way?"

        except litellm.exceptions.BadRequestError as e:
            logger.error(f"LiteLLM BadRequestError: {e.response.text}", exc_info=True)
            return f"I'm experiencing a temporary issue. Please try again shortly."
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return f"Oops! Something unexpected happened. Please try again or contact our support team."


search_kb_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_online_store_kb",
        "description": "Finds information about online product selling topics like shipping, returns, refunds, payment methods, order tracking, product availability, discounts, or how to contact customer support.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The customer's question about an online store product, order, or policy (e.g., 'where's my order', 'how to return', 'what payments do you accept')."}
            },
            "required": ["query"]
        }
    }
}

online_store_chatbot_agent = OnlineStoreChatbotAgent(
    name="OnlineProductSalesChatbot",
    instructions=f"""You are a helpful, concise, and friendly AI chatbot for an online product selling store. Your main goal is to answer common customer queries about orders, products, policies, and assist with general shopping questions.

    **Key Guidelines:**
    1.  **Greeting:** Start with a warm greeting.
    2.  **Concise Answers:** Provide direct, brief answers.
    3.  **Tool Usage:** Use `search_online_store_kb` to find information for customer questions.
    4.  **Direct to Support:** If you can't find a direct answer, or if the query requires specific order details, politely direct the customer to email `support@onlinestore.com` or call `0312-3456789`.
    5.  **No Direct Actions:** You cannot process orders, initiate returns, or access customer accounts. Guide users to relevant policies or human support.
    6.  **Tone:** Professional and approachable.
    """,
    model=MODEL,
    tools_list=[search_kb_tool_schema]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [
        {"role": "system", "content": online_store_chatbot_agent.instructions},
        {"role": "assistant", "content": "Hi there! Welcome to our online store. How can I help you with your shopping today?"}
    ])
    await cl.Message(content="""
**Hi! Welcome to our Online Store.** 👋

I'm your AI assistant, ready to help with quick questions about:
* **Shipping & Delivery**
* **Returns & Refunds**
* **Payment Methods**
* **Order Tracking**
* **Product Availability**
* **Discounts & Promotions**
* **Customer Support**

What can I assist you with today?
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})
    
    try:
        logger.info(f"Processing message: '{message.content}'")
        
        agent_response_content = await online_store_chatbot_agent.generate_response(messages)
        
        messages.append({"role": "assistant", "content": agent_response_content})
        
        await cl.Message(content=agent_response_content).send()

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        error_message = f"""**Oops! Something went wrong.**
Please try asking again. If the issue persists, you can always reach our support team directly:
* **Email:** `support@onlinestore.com`
* **Phone:** `0340-1304435`
"""
        await cl.Message(content=error_message).send()