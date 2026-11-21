import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
load_dotenv()


async def main():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["mathserver.py"],  # Ensure weather.py is also running!
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",  # Ensure server is running here
            "transport": "streamable_http",
        },
    })

    tools = await client.get_tools()
    model = ChatGroq(model="openai/gpt-oss-120b")
    agent = create_agent(
        model,
        tools,
        system_prompt="You are a helpful assistant. Always answer concisely in plain text without LaTeX, backslashes, or markdown formatting.",
    )
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    print("Math response:", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in Islamabad?"}]}
    )
    print("Weather response:", weather_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
