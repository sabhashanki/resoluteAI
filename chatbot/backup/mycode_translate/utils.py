from PyPDF2 import PdfReader
import streamlit as st  
from streamlit_extras.app_logo import add_logo 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

def extract_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


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
    vector_db.save_local("faiss_index")
    # new_db = FAISS.load_local("faiss_index", embeddings)
    st.write('Embeddings generated... Click on the chat button to start the conversation')


def translate_text(text, source='auto', target='hi'):
    return GoogleTranslator(source=source, target=target).translate(text)


# Adds company logo at the top of sidebar
def add_company_logo():
    add_logo('images/resoluteai_logo_smol.jpg', height=80)
    st.markdown(
            """
            <style>
                [data-testid="stSidebarNav"] {
                    padding-top: 1rem;
                    background-position: 10px 10px;
                }
                [data-testid="stSidebarNav"]::before {
                    content: "My Company Name";
                    margin-left: 20px;
                    margin-top: 20px;
                    font-size: 1px;
                    position: relative;
                    top: 1px;
                }
            </style>
            """,
            unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <style>
            .css-1y4p8pa {
                padding-top: 0rem;
                max-width: 50rem;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
    
    
def set_sidebar_state():
    # set sidebar collapsed before login
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'collapsed'

    # hide collapsed control button
    hide_bar = """
            <style>
            [data-testid="collapsedControl"] {visibility:hidden;}
            </style>
            """

    # set sidebar expanded after login
    # if login_after:
    #     st.session_state.sidebar_state = 'expanded'
    # else:
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)