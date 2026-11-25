import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## LangSmith Tracking
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OLLAMA"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question: str, model_name: str, temperature: float) -> str:
    llm = Ollama(model=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})


## Title of the app
st.title("Simple Q&A Chatbot With OLLAMA")

## Sidebar for settings
st.sidebar.title("Settings")
selected_model = st.sidebar.selectbox(
    "Select an Ollama model",
    ["llama3.2", "mistral", "gemma2"],
)
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.7
)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    try:
        with st.spinner("Generating response..."):
            response = generate_response(user_input, selected_model, temperature)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}. Make sure Ollama is running and the model is pulled.")
else:
    st.info("Please provide a query.")