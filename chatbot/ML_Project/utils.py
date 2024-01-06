from deep_translator import GoogleTranslator
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def translate_text(text):
    return GoogleTranslator(source='auto', target='ta').translate(text)

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    st.write('Creating embeddings.... Please wait')
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    st.write('Embeddings generated... Now you can ask questions ?')

    return vector_db

def extract_pdf(pdf_folder):
    text = ""
    for pdf in pdf_folder:
        st.write(f'Extracting {pdf.name}')
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text