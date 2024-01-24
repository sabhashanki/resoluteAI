# import getpass, os,pinecone
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Pinecone
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import TextLoader

# os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key:")
# os.environ["PINECONE_ENV"] = getpass.getpass("Pinecone Environment:")
# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# loader = TextLoader("../../modules/state_of_the_union.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
#     environment=os.getenv("PINECONE_ENV"),  # next to api key in console
# )
# index_name = "langchain-demo"
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
# docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# query = "What did the president say about Ketanji Brown Jackson"
# docs = docsearch.similarity_search(query)
# print(docs[0].page_content)
import time
import tiktoken
import streamlit as st
from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()
# create the length function
def tiktoken_len(text):
	tokenizer = tiktoken.get_encoding('cl100k_base')
	tokens = tokenizer.encode(text,disallowed_special=())
	return len(tokens)

def main():
	uploaded_file=st.file_uploader('Upload File', type='txt')
	if uploaded_file:
		text = ""
		for line in uploaded_file:
			text += str(line, encoding = 'utf-8')
		# st.write(text)
		text_splitter = RecursiveCharacterTextSplitter(
		    chunk_size=400,
		    chunk_overlap=20,
		    length_function=tiktoken_len,
		    separators=["\n\n", "\n", " ", ""]
		)
		chunks = text_splitter.split_text(text)
		
		spec = ServerlessSpec(cloud="aws", region="us-west-2")
		# Create Index
		index_name = 'langchain-retrieval-augmentation'
		pc = Pinecone()
		    pc.create_index(
		        index_name,
		        dimension=1536,  # dimensionality of ada 002
		        metric='dotproduct',
		        spec=spec
		    )

		# initialize the vector store object
		vectorstore = Pinecone(
		    index, embed.embed_query, text_field
		)

		query = "What is the name of the candidate?"

		vectorstore.similarity_search(
		    query,  # our search query
		    k=3  # return 3 most relevant docs
		)


		# completion llm
		llm = ChatOpenAI(
		    openai_api_key=OPENAI_API_KEY,
		    model_name='gpt-3.5-turbo',
		    temperature=0.0
		)

		qa = RetrievalQA.from_chain_type(
		    llm=llm,
		    chain_type="stuff",
		    retriever=vectorstore.as_retriever()
		)

		qa.run(query)


		qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
		    llm=llm,
		    chain_type="stuff",
		    retriever=vectorstore.as_retriever()
		)
		st.write(qa_with_sources(query))
		pc.delete_index(index_name)

if __name__ == '__main__':
	main()





