from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_openai.llms.base import OpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, Qdrant
from utils import translate_text, add_company_logo, lang_select, extract_data, extract_pdf, process_text
import time
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
import configparser
import os

# Generate OpenAI Embeddings and indexing vector DB
def query_answer(query, vector_db):
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response

def load_vector():
	parent_dir = os.getcwd()
	path = os.path.join(parent_dir, user_name)
	if os.path.exists(path):
		if vec_db_name == 'FAISS':
		    vector_db = FAISS.load_local(f"{path}/embeddings/faiss_index", embeddings)
		if vec_db_name == 'CHROMA':
		    vector_db = Chroma(persist_directory=f"{path}/embeddings/chrome_index", embedding_function=embeddings)
	return vector_db

# Generate response for the query        
def chatbox(target):
	vector_db = load_vector()
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
			result = translate_text(query_answer(raw_prompt, vector_db), 'en', target)
			result2 = ""
			for chunk in result.split():
				result2 += chunk + " "
				time.sleep(0.1)
				message_placeholder.markdown(result2 + "▌")
		st.session_state.messages.append({"role": "assistant", "content": result})

st.set_page_config(layout = 'wide')
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
st.title('Chat with PDF')
config = configparser.ConfigParser()
config.read('./config.ini') 
add_company_logo()
vec_db_name = config['VECTOR_DB']['MODEL_NAME']
llm = OpenAI(model_name = 'gpt-3.5-turbo-instruct')
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(llm, chain_type='stuff')

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
		if process_button:
			if doc_type == 'pdf':
				raw_text = extract_pdf(file_folder, user_name)
				process_text(raw_text, user_name)
			else:
				raw_text = extract_data(file_folder, user_name)
				process_text(raw_text, user_name)
	else:
		st.subheader('Login and upload PDFs to access the chat module')

if st.session_state["authentication_status"]:
	chatbox(lang_select(user_lang))