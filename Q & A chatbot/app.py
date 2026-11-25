import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## LangSmith Tracking
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Google GenAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question: str, api_key: str, model_name: str, temperature: float, max_tokens: int) -> str:
    resolved_api_key = api_key.strip() or os.getenv("GOOGLE_API_KEY", "")
    if not resolved_api_key:
        raise ValueError("Please provide a Google GenAI API Key in the sidebar or set GOOGLE_API_KEY in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=resolved_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})


## Title of the app
st.title("Enhanced Q&A Chatbot with Google GenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Google GenAI API Key:", type="password")

## Dropdown to select Gemini models
selected_model = st.sidebar.selectbox(
    "Select a GenAI model",
    ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
)

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=2048, value=512)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    try:
        with st.spinner("Generating response..."):
            response = generate_response(user_input, api_key, selected_model, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please provide a query.")
    