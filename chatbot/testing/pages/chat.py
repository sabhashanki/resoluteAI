from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_openai.llms.base import OpenAI
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from utils import translate_text, add_company_logo, lang_select, extract_data, extract_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
import configparser
import os
import pinecone

index_name="langchainvector"
st.title('Chat with PDF')
config = configparser.ConfigParser()
config.read('./config.ini') 
add_company_logo()
vec_db_name = config['VECTOR_DB']['MODEL_NAME']
llm = OpenAI(model_name = 'gpt-3.5-turbo-instruct')
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(llm, chain_type='stuff')
raw_text = None
# Generate OpenAI Embeddings and indexing vector DB
def query_answer(query, vector_db):
	st.write('Inside query {vector_db}')
	if vec_db_name == 'PINECONE':
		docs = vector_db.similarity_search(query, k=2)
	else:
		docs = vector_db.similarity_search(query)
	response = chain.run(input_documents=docs, question=query)
	return response

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
    st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
    st.write(chunks)
    vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
    st.success('Embeddings generated... Start the conversations', icon="✅")
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
            st.write(chunks)
            vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
            st.write(e)
            st.success('Embeddings generated... Start the conversations', icon="✅")
    except Exception as e:
        st.write(e)
    return vector_db

def load_vector(path):
	try:
		if os.path.exists(path):
			if vec_db_name == 'FAISS':
			    vector_db = FAISS.load_local(f"{path}/embeddings/faiss_index", embeddings)
			if vec_db_name == 'CHROMA':
			    vector_db = Chroma(persist_directory=f"{path}/embeddings/chrome_index", embedding_function=embeddings)
			return vector_db
	except Exception as e:
		st.write('Create Embeddings to load into vector store')

# Generate response for the query        
def chatbox(target, embedding_path):
	global vector_db
	if vec_db_name != 'PINECONE':
		vector_db = load_vector(embedding_path)
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



def main():
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

		user_name = st.session_state["name"]
		parent = os.getcwd()
		path = os.path.join(parent, user_name)
		embedding_path = os.path.join(path, 'embeddings')
		text_path = os.path.join(path, 'extracted_text')
		if process_button:
			global raw_text
			if doc_type == 'pdf':
				raw_text = extract_pdf(file_folder, text_path)
			else:
				raw_text = extract_data(file_folder, text_path)
			vector_db = process_text(raw_text, embedding_path)
			st.write('After Embeddings query {vector_db}')
			target = lang_select(user_lang)
		if vec_db_name == 'FAISS':
			vector_db = load_vector(embedding_path)
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
		with st.sidebar:
			authenticator.logout("Logout", "sidebar")

	else:
		st.subheader('Login and upload PDFs to access the chat module')		

if __name__ == '__main__':
	main()