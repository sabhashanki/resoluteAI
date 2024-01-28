from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone, FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from streamlit_option_menu import option_menu
from deep_translator import GoogleTranslator
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import configparser
import pinecone
load_dotenv()
import time
import yaml
import os

# Initialization
pinecone.init(environment="gcp-starter")
st.set_page_config(layout='centered', page_title='Chat with the Document')
llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
chain = load_qa_chain(llm, chain_type='stuff')
config = configparser.ConfigParser()
vector_config = configparser.ConfigParser()
index_name = "langchainvector"
embeddings = OpenAIEmbeddings()
vector_config.read('./config.ini')
vec_db_name = vector_config['VECTOR_DB']['MODEL_NAME']
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
with open('./config.yaml') as file:
    st.session_state.config = yaml.load(file, Loader=SafeLoader)

st.session_state.authenticator = stauth.Authenticate(
    st.session_state.config['credentials'],
    st.session_state.config['cookie']['name'],
    st.session_state.config['cookie']['key'],
    st.session_state.config['cookie']['expiry_days'],
    st.session_state.config['preauthorized']
)


# access profile section
def profile():
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
        st.subheader(f'Welcome *{st.session_state["name"]}*')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")


# Login Page
def login():
    st.session_state.authenticator.login('Login', location='main')
    if 'login_key' not in st.session_state:
        st.session_state.login_key = None
    if st.session_state["authentication_status"]:
        user_name = st.session_state["name"]
        parent = os.getcwd()
        st.session_state.user_path = os.path.join(parent, user_name)
        if not os.path.exists(st.session_state.user_path):
            os.mkdir(st.session_state.user_path)
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


# Register Page
def register():
    if st.session_state.authenticator.register_user('Registration', preauthorization=False):
        st.success('User registration successfully')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# create onetime password
def forgot_pass():
    username_forgot_pw, email, random_password = st.session_state.authenticator.forgot_password(
        'Create one-time password')
    if username_forgot_pw:
        st.success(f'New random password is : {random_password}.. Change it in next login')
    elif not username_forgot_pw:
        st.error('Username not found')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# update password
def change_pass():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.reset_password(st.session_state["username"], 'Update Password'):
            st.success('New password changed')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to change the password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# update profile info
def update_profile():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.update_user_details(st.session_state["username"], 'Update Profile'):
            st.success('Entries updated successfully')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to update the profile')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# process pdf and create embeddings
def process_text():
    text = ""
    if not os.path.exists(st.session_state.text_path):
        os.mkdir(st.session_state.text_path)
    if st.session_state.doc_type == 'pdf':
        for file in st.session_state.upload_folder:
            pdfdata = PdfReader(file)
            for page in pdfdata.pages:
                text += page.extract_text()
    else:
        for file in st.session_state.upload_folder:
            for line in file:
                text += str(line, encoding='utf-8')
    file = open(st.session_state.text_path + '/' + 'raw_text.txt', 'w')
    file.write(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    try:
        if vec_db_name == 'PINECONE':
            st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
            st.session_state.vector_db = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
            st.success('Embeddings generated... Start the conversations', icon="✅")
        if vec_db_name == 'FAISS':
            st.info('Creating OpenAI embeddings with FAISS.... Please wait', icon="ℹ️")
            st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_db.save_local(f"{st.session_state.embeddings_path}/faiss_index")
            st.success('Embeddings generated... Start the conversations', icon="✅")
    except Exception as e:
        st.write(f'Embedding creation failed : {e}')


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


# final response
def query_answer(query):
    if vec_db_name == 'PINECONE':
        docs = st.session_state.vector_db.similarity_search(query, k=5)
    else:
        docs = st.session_state.vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response


# conversation
def chatbox():
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
                raw_prompt = translate_text(prompt, 'auto', st.session_state.target)
                result = query_answer(prompt)
                result2 = ""
                for chunk in result.split():
                    result2 += chunk + " "
                    time.sleep(0.1)
                    message_placeholder.markdown(result2 + "▌")
            st.session_state.messages.append({"role": "assistant", "content": result})


# document input and upload
def chatpage():
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
            st.session_state.doc_type = st.selectbox('Select document type', (
                'None',
                'pdf',
                'txt',
                'rst',
                'md',
            )
                                                     )
            st.session_state.upload_folder = st.file_uploader("Upload the document", type=['pdf', 'txt', 'md', 'rst'],
                                                              accept_multiple_files=True)
            process_button = st.button("Create Embeddings")
            st.session_state.user_lang = st.selectbox('Select Language', (
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
            st.button('Clear Cache', on_click=st.cache_data.clear())
            if st.session_state["authentication_status"]:
                user_name = st.session_state["name"]
                parent = os.getcwd()
                st.session_state.text_path = os.path.join(parent, user_name)
                st.session_state.embeddings_path = os.path.join(st.session_state.user_path, 'embeddings')
                text_path = os.path.join(st.session_state.user_path, 'extracted_text')
                if process_button:
                    process_text()
            else:
                st.subheader('Login and upload PDFs to access the chat module')
        lang_select()
        chatbox()
    else:
        st.subheader('Login to acesss the chatbot')


# Main function
def main():
    st.sidebar.image('images/resoluteai_logo_smol.jpg')
    with st.sidebar:
        st.session_state.key = option_menu("ChatBot", ["Home", 'Profile', 'Chat'],
                                           icons=['house', 'unlock', 'chat'], menu_icon="chat", orientation='vertical')
    if st.session_state.key == 'Home':
        home()
    if st.session_state.key == 'Chat':
        chatpage()
    if st.session_state.key == 'Profile':
        profile()


if __name__ == "__main__":
    main()
    # st.cache_data.clear()
