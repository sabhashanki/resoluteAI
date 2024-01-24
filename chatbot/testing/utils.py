import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
import pinecone, pickle, os, configparser
import streamlit_authenticator as stauth
from streamlit_extras.app_logo import add_logo 
from langchain_openai import OpenAIEmbeddings
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS, Pinecone

# Initialization
config = configparser.ConfigParser()
config.read('./config.ini') 
index_name="langchainvector"

def extract_pdf(pdf_folder, path):
    text = ""
    for pdf in pdf_folder:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    if not os.path.exists(path):
        os.mkdir(path)
    file = open(path + '/' + 'raw_text.txt' , 'w')
    file.write(text)
    return text


def extract_data(folder, path):
    text = ""
    for txt in folder:
        for line in txt:
            text += str(line, encoding = 'utf-8')
    if not os.path.exists(path):
        os.mkdir(path)
    file = open(path + '/' + 'raw_text.txt' , 'w')
    file.write(text)
    return text


def process_text(text, path):
    pinecone.init(api_key="db6b2a8c-d59e-48e1-8d5c-4c2704622937",environment="gcp-starter")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vec_db_name = config['VECTOR_DB']['MODEL_NAME']
    try:
        if vec_db_name == 'FAISS':
            st.info('Creating OpenAI embeddings with FAISS.... Please wait', icon="ℹ️")
            vector_db = FAISS.from_texts(chunks, embeddings)
            vector_db.save_local(f"{path}/embeddings/faiss_index")
            st.success('Embeddings generated... Start the conversations', icon="✅")

        if vec_db_name == 'CHROMA':
            st.info('Creating OpenAI embeddings with CHROMA.... Please wait', icon="ℹ️")
            vector_db = Chroma.from_texts(chunks, embeddings, persist_directory = f"{path}/embeddings/chrome_index")
            st.success('Embeddings generated... Start the conversations', icon="✅")
            vector_db=Pinecone.from_texts(chunks,embeddings,index_name=index_name)
            st.success('Embeddings generated... Start the conversations', icon="✅")

        if vec_db_name == 'PINECONE':
            st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
            vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
            st.success('Embeddings generated... Start the conversations', icon="✅")
    except Exception as e:
        st.write(e)

    return vector_db

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
                    margin-left: 0px;
                    margin-top: 0px;
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
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)

def login():
    # Reading login information
    user_info = {}
    cred_path = Path(__file__).parent / "./hashed_passwords.pkl"
    with cred_path.open('rb') as file:
        user_info = pickle.load(file)

    credentials = {
        'usernames' : {
            user_info['usernames'][0] : {
                'name' : user_info['names'][0],
                'password' : user_info['passwords'][0]
            }
        }
    }
    cookie_name = 'sample_app'
    authenticator = stauth.Authenticate(credentials, cookie_name, 'abcd', cookie_expiry_days=60)

    st.session_state['authenticator'] = authenticator
    st.session_state['cookie_name'] = cookie_name
    name, authentication_status, username = authenticator.login("Login", "main")
    return authentication_status, authenticator

def logout():
    login.authenticator.cookie_manager.delete(login.cookie_name)
    st.session_state['logout'] = True
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.session_state['authentication_status'] = None

hide_bar = '''
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
'''

def lang_select(user_lang):
    lang = {
'Tamil':  'ta',
'English': 'en',
'Hindi':   'hi',
'Marathi':   'mr',
'Malayalam':   'ml',
'Kannada':   'ka',
'Telugu':   'tl',
'Assamese':   'as',
'Gujarati':   'gu',
'Oriya':   'or',
'Punjabi':   'pa',
'Bengali':   'bn',
'Spanish':   'es',
'Urdu':   'ur',
'Sanskrit' : 'sa',
'Chinese(simplified)': 'zh-CN',
'French':   'fr',
'Korean':   'ko',
'Japanese':   'ja',
'Portuguese':   'pt',
'Italian':   'it',
'Russian':   'ru'
}
    for key, value in lang.items():
        if user_lang == key:
            return value