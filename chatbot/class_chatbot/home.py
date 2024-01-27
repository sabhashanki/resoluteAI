from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_option_menu import option_menu
from deep_translator import GoogleTranslator
from langchain.vectorstores import Pinecone
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import OpenAI
from PyPDF2 import PdfReader
import streamlit as st
import configparser
import langchain
load_dotenv()
import pinecone
import openai
import time
import yaml
import os

# Initialization
pinecone.init(environment="gcp-starter")
st.set_page_config(layout = 'centered', page_title = 'Chat with the Document')
llm = OpenAI(model_name = 'gpt-3.5-turbo-instruct')
chain = load_qa_chain(llm, chain_type='stuff')
config = configparser.ConfigParser()
embeddings = OpenAIEmbeddings()
index_name="langchainvector"
config.read('./config.ini') 
vec_db_name = config['VECTOR_DB']['MODEL_NAME']

# Home Page
def home():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    st.title("This is my Home page")
    if not st.session_state["authentication_status"]:
        st.write('Not Logged in')
    if st.session_state["authentication_status"]:
        st.write('Already logged in')
        with st.sidebar:
            authenticator.logout("Logout", "sidebar")

    # if st.session_state.login_clicked:
    #     st.switch_page('login')

    
# Login Page
def login():
    st.session_state.login_clicked = True
    st.title("Login page")
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    authenticator.login('Login', location = 'main')
    if st.session_state["authentication_status"]:
        st.title(f'Welcome *{st.session_state["name"]}*')
        st.subheader('Click on the Chat to upload document and access AI chatbot')
        user_name = st.session_state["name"]
        parent = os.getcwd()
        st.session_state.user_path = os.path.join(parent, user_name)
        if not os.path.exists(st.session_state.user_path):
            os.mkdir(path)
        with st.sidebar:
               authenticator.logout("Logout", "sidebar")
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


# Register Page
def register():
    st.title("Register page")
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    if authenticator.register_user(preauthorization=False):
        st.success('User registration successfully')
    with open('./config.yaml', 'a') as file:
        yaml.dump(config, file, default_flow_style=False)


def forgot_pass():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    username_forgot_pw, email, random_password = authenticator.forgot_password('Forgot password')
    if username_forgot_pw:
        st.success(f'New random password is : {random_password}.. Change it in next login')
    elif username_forgot_pw == False:
        st.error('Username not found')
    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def change_pass():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    if st.session_state["authentication_status"]:
        if authenticator.reset_password(st.session_state["username"], 'Reset password'):
            st.success('New password changed')
        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to change the password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def update_profile():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    if st.session_state["authentication_status"]:
        if authenticator.update_user_details(st.session_state["username"], 'Update user details'):
            st.success('Entries updated successfully')
        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to update the profile')
    with open('./config.yaml', 'a') as file:
        yaml.dump(config, file, default_flow_style=False)


def process_text():
    text = ""
    if not os.path.exists(st.session_state.txt_path):
        os.mkdir(st.session_state.txt_path)
    if st.session_state.doc_type  == 'pdf':
        for file in st.session_state.upload_folder:
            pdfdata = PdfReader(file)
            for page in pdfdata.pages:
                text += page.extract_text()
    else:
        for file in st.session_state.upload_folder:
            for line in file:
                text += str(line, encoding = 'utf-8')
    file = open(st.session_state.txt_path + '/' + 'raw_text.txt' , 'w')
    file.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    try:
        if vec_db_name == 'PINECONE':
            st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
            st.session_state.vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
            st.success('Embeddings generated... Start the conversations', icon="✅")
        if vec_db_name == 'FAISS':
            st.info('Creating OpenAI embeddings with FAISS.... Please wait', icon="ℹ️")
            vector_db = FAISS.from_texts(chunks, embeddings)
            vector_db.save_local(f"{embeddings_path}/faiss_index")
            st.success('Embeddings generated... Start the conversations', icon="✅")
    except Exception as e:
        st.write(f'Embedding creation failed : {e}')



def translate_text(text, source='auto', target='ta'):
    return GoogleTranslator(source=source, target=target).translate(text)


def lang_select():
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
        if st.session_state.user_lang == key:
            st.session_state.target = value
            

def query_answer(query):
    if vec_db_name == 'PINECONE':
        docs = st.session_state.vector_db.similarity_search(query, k=5)
    else:
        docs = st.session_state.vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response


def chatbox():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if prompt := st.chat_input('Ask question about PDF content'):
        st.session_state.messages.append({'role' : 'user', 'content' : prompt})
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


def chatpage():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")
        st.session_state.doc_type = st.selectbox('Select document type', (
                    'None',
                    'pdf', 
                    'txt',
                    'rst',
                    'md',
                    )
                )
        st.session_state.upload_folder = st.file_uploader("Upload the document",type=['pdf','txt','md','rst'],accept_multiple_files=True)           
        process_button=st.button("Create Embeddings")
        st.session_state.user_lang =st.selectbox('Select Language', (
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


def main():
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
    st.sidebar.image('images/resoluteai_logo_smol.jpg')
    pages = {
        "Home": home,
        "Login": login,
        "Chat" : chatpage, 
        "Register" : register,
        "Forgot Password" : forgot_pass,
        "Reset Password" : change_pass,
        "Update Profile" : update_profile
    }
    st.sidebar.title("ResoluteAI Software Chatbot")
    page = st.sidebar.selectbox("Browse Pages", tuple(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()