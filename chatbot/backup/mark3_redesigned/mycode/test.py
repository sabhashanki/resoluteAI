from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain_openai.llms.base import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, Qdrant
from deep_translator import GoogleTranslator
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from PyPDF2 import PdfReader
import configparser, os, yaml, time
from dotenv import load_dotenv

# Initialization
load_dotenv()
config = configparser.ConfigParser()
config.read('./config.ini') 
vec_db_name = config['VECTOR_DB']['MODEL_NAME']
llm = OpenAI(model_name = 'gpt-3.5-turbo-instruct')
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(llm, chain_type='stuff')
vector_db = None
st.set_page_config(layout = 'centered', page_title = 'Chat with the Document')

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
    authenticator.login('Login', 'main')
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
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vec_db_name = config['VECTOR_DB']['MODEL_NAME']

    if vec_db_name == 'FAISS':
        st.info('Creating OpenAI embeddings with FAISS.... Please wait', icon="ℹ️")
        vector_db = FAISS.from_texts(chunks, embeddings)
        vector_db.save_local(f"{path}/faiss_index")

    if vec_db_name == 'CHROMA':
        st.info('Creating OpenAI embeddings with CHROMA.... Please wait', icon="ℹ️")
        vector_db = Chroma.from_texts(chunks, embeddings, persist_directory = f"{path}/chrome_index")
    st.success('Embeddings generated... Start the conversations', icon="✅")


def translate_text(text, source='auto', target='hi'):
    return GoogleTranslator(source=source, target=target).translate(text)

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

def query_answer(query):
    global vector_db
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response

def chatbox(target):
    st.subheader('Chat with the Document !!')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
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
            result = translate_text(query_answer(raw_prompt), 'en', target)
            result2 = ""
            for chunk in result.split():
                result2 += chunk + " "
                time.sleep(0.1)
                message_placeholder.markdown(result2 + "▌")
        st.session_state.messages.append({"role": "assistant", "content": result})

def chatpage():
    global vector_db
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
        doc_type = st.selectbox('Select document type', (
                    'None',
                    'pdf', 
                    'txt',
                    'rst',
                    'md',
                    )
                )
        file_folder = st.file_uploader("Upload the document",type=['pdf','txt','md','rst'],accept_multiple_files=True)           
        process_button=st.button("Create Embeddings")
        user_lang =st.selectbox('Select Language', (
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
            path = os.path.join(parent, user_name)
            embedding_path = os.path.join(path, 'embeddings')
            text_path = os.path.join(path, 'extracted_text')
            if process_button:
                if doc_type == 'pdf':
                    raw_text = extract_pdf(file_folder, text_path)
                    process_text(raw_text, embedding_path)
                else:
                    raw_text = extract_data(file_folder, text_path)
                    process_text(raw_text, embedding_path)
            try:
                if vec_db_name == 'FAISS':
                    vector_db = FAISS.load_local(f'{embedding_path}/faiss_index', embeddings)
                if vec_db_name == 'CHROMA':
                    vector_db = Chroma(persist_directory = f'{embedding_path}/chrome_index',
                    embedding_function = embeddings )
            except Exception as e:
                st.write('Upload files to chat')
                
        else:
            st.subheader('Login and upload PDFs to access the chat module')
    chatbox(lang_select(user_lang))



def main():
    st.sidebar.image('images/resoluteai_logo_smol.jpg')
    pages = {
        "Home": home,
        "Chat" : chatpage, 
        "Login": login,
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