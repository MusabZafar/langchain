import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("AGENTROUTER_API_KEY")

llm = ChatAnthropic(
    api_key=api_key,
    base_url="https://agentrouter.org",
    model="gpt-5.6-sol",
    default_headers={
        "User-Agent": "claude-cli/1.0.0 (external, cli)",
        "x-app": "cli",
        "anthropic-beta": "claude-code-20250219",
    },
)

print("AgentRouter Chat Ready! Type 'exit' to quit.\n")

while True:
    user_query = input("Your Query: ")
    if user_query.strip().lower() in ["exit", "quit"]:
        break
    if not user_query.strip():
        continue

    try:
        response = llm.invoke(user_query)
        print("\n--- RESPONSE ---")
        print(response.content)
        print("-" * 40 + "\n")
    except Exception as e:
        print("\n--- ERROR DETAILS ---")
        print(e)
