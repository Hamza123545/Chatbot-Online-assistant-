import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-1.5-pro"

@function_tool
def check_ticket_status(ticket_id: str) -> str:
    return f"Ticket {ticket_id} is being reviewed. We'll update you soon."

agent = Agent(
    name="CustomerSupportAssistant",
    instructions="You are a polite and empathetic customer support assistant...",
    model=LitellmModel(model=MODEL, api_key=API_KEY, api_base=BASE_URL, provider="google"),
    tools=[check_ticket_status]
)

async def main():
    result = await Runner.run(agent, "Can you check ticket #12345?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
