from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import OpenAI
from PyPDF2 import PdfReader
import streamlit as st
import langchain
import pinecone
import openai
import os

# Initialization
pinecone.init(api_key="db6b2a8c-d59e-48e1-8d5c-4c2704622937",environment="gcp-starter")
llm=OpenAI(model_name="gpt-3.5-turbo-instruct")
chain=load_qa_chain(llm,chain_type="stuff")
index_name="langchainvector"
load_dotenv()


# Translatiton
def translate_text(text, source='auto', target='hi'):
    return GoogleTranslator(source=source, target=target).translate(text)


## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results


# Extract document and create embeddings
def process_text(pdf_folder, doc_type, path):
	text = ""
	if not os.path.exists(path):
		os.mkdir(path)
	for file in pdf_folder:
		if doc_type == 'pdf':
			pdfdata = PdfReader(file)
			for page in pdfdata:
				text += page.extract_text()
		else:
			for line in file:
				text += str(line, encoding = 'utf-8')
	file = open(path + '/' + 'raw_text.txt' , 'w')
	file.write(text)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000,
		chunk_overlap=100, 
		length_function=len
	)
	chunks = text_splitter.split_text(text)
	embeddings = OpenAIEmbeddings()
	st.info('Creating OpenAI embeddings with PINECONE.... Please wait', icon="ℹ️")
	vector_db = Pinecone.from_texts(chunks,embeddings,index_name=index_name)
	st.success('Embeddings generated... Start the conversations', icon="✅")
	return vector_db



def query_answer(query, vector_db):
	docs = vector_db.similarity_search(query, k=2)
	response = chain.run(input_documents=docs, question=query)
	return response


def chatbox():
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
			result = translate_text(query_answer(raw_prompt, vector_db), 'en', 'ta')

			result2 = ""
			for chunk in result.split():
				result2 += chunk + " "
				time.sleep(0.1)
				message_placeholder.markdown(result2 + "▌")
		st.session_state.messages.append({"role": "assistant", "content": result})


def main():
	parent = os.getcwd()
	txt_path = os.path.join(parent, 'extract_text')
	doc_type = st.selectbox('Document type', ('None','PDF','TXT'), 'RST','MD')
	file_uploader = st.file_uploader('Upload files', type = ['pdf', 'txt', 'rst','md'])
	submitBtn = st.button('Submit', on_click = process_text(file_uploader, doc_type, txt_path)
	if file_uploader:
		raw_text = extract_pdf(file_uploader)
		vector_db = process_text(raw_text)
		chatbox()




