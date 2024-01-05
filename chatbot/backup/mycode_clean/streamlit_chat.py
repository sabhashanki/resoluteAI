import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from deep_translator import GoogleTranslator


# Load environment variables
load_dotenv()

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
    print('Creating embeddings.... Please wait')
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    print('Embeddings generated... Now you can ask questions ?')

    return new_db


st.title("Shankesh AI - Chat with your PDF 💬")

pdf = st.file_uploader('Upload your PDF Document', type='pdf')

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    vector_db = process_text(text) 
    
llm = OpenAI()
chain = load_qa_chain(llm, chain_type='stuff')

def query_answer(query):
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response
        
prompt = st.chat_input("Ask question about PDF content")
if prompt:
	st.write(f'USER : {translate_text(prompt)}')
	result = query_answer(prompt)
	st.write(f'SHANKESH AI: {translate_text(result)}')