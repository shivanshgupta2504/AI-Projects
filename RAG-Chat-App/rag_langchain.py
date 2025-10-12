import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

# Loading the Documents
doc_paths = os.listdir("docs")

docs = []
for doc_file in doc_paths:
    file_path = Path("docs", doc_file)

    try:
        if file_path.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.name.endswith(".txt") or file_path.name.endswith(".md"):
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file format: {file_path.name}")
            continue

        docs.extend(loader.load())

    except Exception as e:
        print(f"Error Loading document {file_path.name}: {e}")

