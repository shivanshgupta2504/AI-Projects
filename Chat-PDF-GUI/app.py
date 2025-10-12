from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

def main():
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF 💬")

    prompt = """
    This is context:
    {context}
    
    This is my question:
    {query}
    """

    # Upload the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract text from file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User Input
        user_question = st.text_input("Ask a Question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = ChatOpenAI()

            qa_prompt = ChatPromptTemplate.from_messages([("human", prompt)])
            chain = load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=qa_prompt)
            with get_openai_callback() as cb:
                response = chain.invoke({
                    "input_documents": docs,
                    "query": user_question
                })
                print(cb)

            st.write(response.get("output_text", ""))

if __name__ == "__main__":
    main()