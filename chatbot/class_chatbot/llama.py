from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import OpenAI, HuggingFaceLLM
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

# response = OpenAI().complete('Daniel Craig is a ')
# print(response)

prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided

"""

llm = HuggingFaceLLM(
    model_name='/home/ms/resoluteAI/chatbot/class_chatbot/paraphrase-MiniLM-L6-v2',
    tokenizer_name='/home/ms/resoluteAI/chatbot/class_chatbot/paraphrase-MiniLM-L6-v2',
    system_prompt=prompt,
    device_map='cpu'
)
# embed = HuggingFaceEmbeddings(model_name='/home/ms/resoluteAI/chatbot/class_chatbot/paraphrase-MiniLM-L6-v2')
# embed_model = LangchainEmbedding(embed)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)


# llm = OpenAI(temperature=0.1, model='gpt-3.5-turbo-instruct')
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)

documents = SimpleDirectoryReader('/home/ms/resoluteAI/chatbot/class_chatbot/data').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir='./data/embeddings/')
storage_context = StorageContext.from_defaults(persist_dir='./data/embeddings/')
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query('who are all the signing parties ?')
print(response)