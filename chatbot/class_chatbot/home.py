from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone, FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from streamlit_option_menu import option_menu
from deep_translator import GoogleTranslator
import streamlit_authenticator as stauth
from streamlit.logger import get_logger
from streamlit_extras.grid import grid
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import configparser
import pinecone
import time
import yaml
import os


# Initialization
load_dotenv()
logger = get_logger(__name__)
# config = configparser.ConfigParser()
init_config = configparser.ConfigParser()
index_name = "langchainvector"
init_config.read('./config.ini')
vec_db_name = init_config['VECTOR_DB']['MODEL_NAME']
embed_model = init_config['EMBEDDING_MODEL']['MODEL_NAME']
st.set_page_config(layout='centered', page_title='Chat with the Document')

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'text_path' not in st.session_state:
    st.session_state.text_path = None
if 'embeddings_path' not in st.session_state:
    st.session_state.embeddings_path = None
if 'user_lang' not in st.session_state:
    st.session_state.user_lang = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'doc_type' not in st.session_state:
    st.session_state.doc_type = None
if 'upload_folder' not in st.session_state:
    st.session_state.upload_folder = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_path' not in st.session_state:
    st.session_state.user_path = []
if 'key' not in st.session_state:
    st.session_state.key = None
if 'authenticator' not in st.session_state:
    st.session_state.authenticator = False
if 'config' not in st.session_state:
    st.session_state.config = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = False
if 'chain' not in st.session_state:
    st.session_state.chain = False

logger.info('Initialization Done')


@st.cache_data(persist='disk',experimental_allow_widgets=True)
def initFunc(embed_model):
    logger.info('Initialization inside function')
    pinecone.init(environment="gcp-starter")
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
    st.session_state.chain = load_qa_chain(llm, chain_type='stuff')
    if embed_model == 'OPENAI':
        st.session_state.embeddings = OpenAIEmbeddings()
        logger.info('Using OpenAI embeddings')
    if embed_model == 'HUGGINGFACE':
        logger.info('Using Huggingface embeddings')
        st.session_state.embeddings = HuggingFaceEmbeddings(
    model_name="./paraphrase-MiniLM-L6-v2/",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


# access profile section
def profile():
    logger.info('accessing profile section')
    if st.session_state["authentication_status"]:
        profile_menu = option_menu(None, ["Update Password", "Update Profile"],
                                   icons=['unlock', 'lock'],
                                   orientation="horizontal", key='menu_profile')
        if profile_menu == 'Update Password':
            change_pass()
        if profile_menu == 'Update Profile':
            update_profile()
    if not st.session_state["authentication_status"]:
        st.subheader('Login to access the profile')


# Home Page
def home():
    logger.info('accessing profile section')
    if not st.session_state["authentication_status"]:
        home_menu = option_menu(None, ["Login", "Register", "Forgot Password"],
                                icons=['unlock', 'lock', 'lock'],
                                orientation="horizontal", key='home_menu')
        if home_menu == 'Login':
            login()
        if home_menu == 'Register':
            register()
        if home_menu == 'Forgot Password':
            forgot_pass()

    if st.session_state["authentication_status"]:
        st.subheader(f'Welcome *{st.session_state["name"]}*, Click on the chat !!')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")


# Login Page
def login():
    logger.info('accessing login section')
    st.session_state.authenticator.login('Login', location='main')
    if 'login_key' not in st.session_state:
        st.session_state.login_key = None
    if st.session_state["authentication_status"]:
        logger.info('Login authentication successful')
        user_name = st.session_state["name"]
        parent = os.getcwd()
        st.session_state.user_path = os.path.join(parent, user_name)
        if not os.path.exists(st.session_state.user_path):
            os.mkdir(st.session_state.user_path)
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        logger.info('Incorrect password/username')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


# Register Page
def register():
    if st.session_state.authenticator.register_user('Registration', preauthorization=False):
        st.success('User registration successfully')
        logger.info('Registration successful')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# create onetime password
def forgot_pass():
    username_forgot_pw, email, random_password = st.session_state.authenticator.forgot_password(
        'Create one-time password')
    if username_forgot_pw:
        st.success(f'New random password is : {random_password}.. Change it in next login')
        logger.info(f'Onetime password generated for the user {username_forgot_pw}: {random_password}')
    elif not username_forgot_pw:
        st.warning('Enter the username')
        logger.info('Invalid username for generating onetime password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('onetime password updated in yaml file')


# update password
def change_pass():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.reset_password(st.session_state["username"], 'Update Password'):
            st.success('New password changed')
            logger.info('user password changed')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to change the password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('new password updated in yaml file')


# update profile info
def update_profile():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.update_user_details(st.session_state["username"], 'Update Profile'):
            st.success('Entries updated successfully')
            logger.info('user info changed successfully')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to update the profile')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('user info updated in yaml file')


# process pdf and create embeddings
# @st.cache_data(persist='disk')
def process_text():
    text = ""
    if not os.path.exists(st.session_state.text_path):
        logger.info('text_path not created.. creating inside process_text module')
        os.mkdir(st.session_state.text_path)
    pdf_pages = 0
    if st.session_state.doc_type == 'pdf':
        for file in st.session_state.upload_folder:
            pdfdata = PdfReader(file)
            for page in pdfdata.pages:
                pdf_pages += 1
                text += page.extract_text()
            logger.info('Text extracted from the pdf')
            st.info(f'PDF count: {len(st.session_state.upload_folder)}, Extracted pages: {pdf_pages}')
            logger.info(f'PDF count: {len(st.session_state.upload_folder)}, page count: {pdf_pages}')
    else:
        for file in st.session_state.upload_folder:
            for line in file:
                text += str(line, encoding='utf-8')
            logger.info(f'Document count: {len(st.session_state.upload_folder)}')
    file = open(st.session_state.text_path + '/' + 'raw_text.txt', 'w')
    file.write(text)
    logger.info('Text extracted and saved in text file')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info(f'text split into chunks : {len(chunks)}')
    try:
        if vec_db_name == 'PINECONE':
            st.info(f'Creating {embed_model} embeddings with PINECONE.... Please wait', icon="ℹ️")
            st.session_state.vector_db = Pinecone.from_texts(chunks, st.session_state.embeddings, index_name=index_name)
            st.success('Embeddings generated... Start the conversations', icon="✅")
            logger.info('Embeddings created and saved in the pinecone vector database')
        if vec_db_name == 'FAISS':
            st.info(f'Creating {embed_model} embeddings with FAISS.... Please wait', icon="ℹ️")
            st.session_state.vector_db = FAISS.from_texts(chunks, st.session_state.embeddings)
            st.session_state.vector_db.save_local(f"{st.session_state.embeddings_path}/faiss_index")
            st.success('Embeddings generated... Start the conversations', icon="✅")
            logger.info('Embeddings created and saved in the FAISS vector store')
    except Exception as e:
        st.write(f'Clear the cache and run again')
        logger.info(f'Embedding creation failed: {e}')


# google translate
def translate_text(text, source='auto', target='ta'):
    return GoogleTranslator(source=source, target=target).translate(text)


# output language select
def lang_select():
    lang = {
        'Tamil': 'ta',
        'English': 'en',
        'Hindi': 'hi',
        'Marathi': 'mr',
        'Malayalam': 'ml',
        'Kannada': 'ka',
        'Telugu': 'tl',
        'Assamese': 'as',
        'Gujarati': 'gu',
        'Oriya': 'or',
        'Punjabi': 'pa',
        'Bengali': 'bn',
        'Spanish': 'es',
        'Urdu': 'ur',
        'Sanskrit': 'sa',
        'Chinese(simplified)': 'zh-CN',
        'French': 'fr',
        'Korean': 'ko',
        'Japanese': 'ja',
        'Portuguese': 'pt',
        'Italian': 'it',
        'Russian': 'ru'
    }
    for key, value in lang.items():
        if st.session_state.user_lang == key:
            st.session_state.target = value
            logger.info(f'Language selected: {value}')


# final response
def query_answer(query):
    if vec_db_name == 'PINECONE':
        docs = st.session_state.vector_db.similarity_search(query, k=5)
        logger.info('pinecone vector db chosen for similarity search')
    else:
        docs = st.session_state.vector_db.similarity_search(query)
        logger.info('FAISS vector db chosen for similarity search')
    response = st.session_state.chain.run(input_documents=docs, question=query)
    return response


# conversation
def chatbox():
    logger.info('Accessing chatbox')
    if st.session_state["authentication_status"]:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        if prompt := st.chat_input('Ask question about PDF content'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                raw_prompt = translate_text(prompt, 'auto', 'en')
                logger.info('Input text translated into english')
                result = translate_text(query_answer(raw_prompt), 'auto', st.session_state.target)
                result2 = ""
                for chunk in result.split():
                    result2 += chunk + " "
                    time.sleep(0.1)
                    message_placeholder.markdown(result2 + "▌")
            st.session_state.messages.append({"role": "assistant", "content": result})


# document input and upload
def chatpage():
    logger.info('Accessing chatpage')
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
            input_grid = grid(2, vertical_align="bottom")
            st.session_state.doc_type = input_grid.selectbox('Select document type', (
                'None',
                'pdf',
                'txt',
                'rst',
                'md',
            ))
            st.session_state.user_lang = input_grid.selectbox('Select Language', (
                'English',
                'Tamil',
                'Hindi',
                'Malayalam',
                'Kannada',
                'Telugu',
                'Marathi',
                'Assamese',
                'Bengali',
                'Gujarati',
                'Konkani',
                'Oriya',
                'Punjabi',
                'Sanskrit',
                'Urdu',
                'Chinese(simplified)',
                'French',
                'Korean',
                'Japanese',
                'Portuguese',
                'Italian',
                'Russian'
            ))
            st.session_state.upload_folder = st.file_uploader("Upload the document", type=['pdf', 'txt', 'md', 'rst'],
                                                              accept_multiple_files=True)
            my_grid = grid(2, vertical_align="bottom")
            process_button = my_grid.button("Start Processing")
            cacheBtn = my_grid.button('Clear Cache')
            if cacheBtn:
                st.cache_data.clear()
                st.success('Cached files cleared !!')
            if st.session_state["authentication_status"]:
                user_name = st.session_state["name"]
                parent = os.getcwd()
                st.session_state.text_path = os.path.join(parent, user_name)
                st.session_state.embeddings_path = os.path.join(st.session_state.user_path, 'embeddings')
                st.session_state.text_path = os.path.join(st.session_state.user_path, 'extracted_text')
                if process_button:
                    process_text()
            else:
                st.subheader('Login and upload PDFs to access the chat module')
        lang_select()
        chatbox()
    else:
        st.subheader('Login to access the chat bot')


# Main function
def main():
    logger.info('Accessing main function')
    initFunc(embed_model)
    with open('./config.yaml') as file:
        st.session_state.config = yaml.load(file, Loader=SafeLoader)

    st.session_state.authenticator = stauth.Authenticate(
        st.session_state.config['credentials'],
        st.session_state.config['cookie']['name'],
        st.session_state.config['cookie']['key'],
        st.session_state.config['cookie']['expiry_days'],
        st.session_state.config['preauthorized']
    )
    with st.sidebar:
        st.image('images/resoluteai_logo_smol.jpg')
        st.session_state.key = option_menu("Document Chat", ["Home", 'Profile', 'Chat'],
                                           icons=['house', 'unlock', 'chat'], orientation='horizontal')
    if st.session_state.key == 'Home':
        home()
    if st.session_state.key == 'Chat':
        chatpage()
    if st.session_state.key == 'Profile':
        profile()


if __name__ == "__main__":
    main()

