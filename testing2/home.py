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
import langchain
load_dotenv()
import pinecone
import openai
import time
import yaml
import os

# Initialization
pinecone.init(api_key="db6b2a8c-d59e-48e1-8d5c-4c2704622937",environment="gcp-starter")
llm=OpenAI(model_name="gpt-3.5-turbo-instruct")
chain=load_qa_chain(llm,chain_type="stuff")
index_name="langchainvector"


# Home Page
def home():
    st.title("This is my Home page")


# Login Page
def login():
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
        path = os.path.join(parent, user_name)
        if not os.path.exists(path):
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
    if authenticator.register_user('Register user', preauthorization=False):
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
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to update the profile')
    with open('./config.yaml', 'a') as file:
        yaml.dump(config, file, default_flow_style=False)


# Translatiton
def translate_text(text, source='auto', target='hi'):
    return GoogleTranslator(source=source, target=target).translate(text)


# Extract document and create embeddings
def process_text():
	text = ""
	if not os.path.exists(st.session_state.txt_path):
		os.mkdir(st.session_state.txt_path)
	if st.session_state.doc_type  == 'PDF':
		for file in st.session_state.upload_folder:
			pdfdata = PdfReader(file)
			for page in pdfdata.pages:
				text += page.extract_text()
	else:
		for file in pdf_folder:
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
	st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
	st.session_state.vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
	st.success('Embeddings generated... Start the conversations', icon="✅")



def query_answer(query):
	docs = st.session_state.vector_db.similarity_search(query, k=2)
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
			raw_prompt = translate_text(prompt, 'auto', 'en')
			result = query_answer(prompt)
			result2 = ""
			for chunk in result.split():
				result2 += chunk + " "
				time.sleep(0.1)
				message_placeholder.markdown(result2 + "▌")
		st.session_state.messages.append({"role": "assistant", "content": result})


def about(key):
	selection = st.session_state[key]
	if selection == 'Home':
		home()
	if selection == 'Login':
		login()
	if selection == 'Register':
		register()
	if selection == 'Forgot Password':
		forgot_pass()

def tasks():
	st.write('Tasks')

def main():
	if 'vector_db' not in st.session_state:
		st.session_state.vector_db = None
	if 'txt_path' not in st.session_state:
		st.session_state.txt_path = None
	if 'doc_type' not in st.session_state:
		st.session_state.doc_type = None
	if 'upload_folder' not in st.session_state:
		st.session_state.upload_folder = None
	if 'messages' not in st.session_state:
		st.session_state.messages = []
	st.session_state.txt_path = os.path.join(os.getcwd(), 'extract_text')
	with st.sidebar:
		selected5 = option_menu(None, ["Home", "Login", "Register", 'Forgot Passoword'],
	                        icons=['house', 'login', "register", 'gear'],
	                        on_change=about, key='menu_5', orientation="vertical")
		st.session_state.doc_type = st.selectbox('Document type', ('None','PDF','TXT', 'RST','MD'))
		st.session_state.upload_folder = st.file_uploader('Upload files', type = ['pdf', 'txt', 'rst','md'], accept_multiple_files=True)
		submitBtn = st.button('Submit') 
		if submitBtn:
			process_text()
		
	chatbox()


if __name__ == '__main__':
	main()
