import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load .env from current folder or parent folder
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

agentrouter_api_key = os.getenv("AGENTROUTER_API_KEY")

# Working AgentRouter LLM configuration
llm = ChatAnthropic(
    api_key=agentrouter_api_key,
    base_url="https://agentrouter.org",
    model="gpt-5.6-sol",
    default_headers={
        "User-Agent": "claude-cli/1.0.0 (external, cli)",
        "x-app": "cli",
        "anthropic-beta": "claude-code-20250219",
    },
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

st.title("RAG Document Q&A With AgentRouter")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    with st.spinner("Creating vector database..."):
        create_vector_embedding()
    st.success("Vector Database is ready")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first to load your PDFs.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time: {time.process_time() - start}")

        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("------------------------")
