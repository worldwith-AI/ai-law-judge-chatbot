import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Indian Law Judge Bot", layout="wide")
st.title("⚖️ Indian Law Judge Chatbot")

# -----------------------------
# 1. Web Scraping
# -----------------------------
def scrape_law_data():
    urls = [
        "https://indiankanoon.org/doc/1569253/",
        "https://www.indiacode.nic.in/"
    ]

    full_text = ""

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")

            for p in paragraphs:
                full_text += p.get_text() + "\n"

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return full_text


# -----------------------------
# 2. Vector DB
# -----------------------------
@st.cache_resource
def create_vector_db():

    law_text = scrape_law_data()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([law_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_db = FAISS.from_documents(docs, embeddings)

    return vector_db


# -----------------------------
# 3. Load Groq LLM
# -----------------------------
@st.cache_resource
def load_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",   # fast model
        temperature=0.3
    )

    return llm


# -----------------------------
# 4. Main App
# -----------------------------
query = st.text_input("Ask your legal question:")

if query:

    with st.spinner("Analyzing law sections..."):

        vector_db = create_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        llm = load_llm()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": query})

        st.subheader("⚖️ Judge AI Response:")
        st.write(result["result"])