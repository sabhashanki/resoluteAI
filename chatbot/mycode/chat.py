from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS, Chroma, Qdrant
from utils import translate_text, add_company_logo
import time
import configparser

# Initialization
config = configparser.ConfigParser()
config.read('./config.ini') 
vec_db_name = config['VECTOR_DB']['MODEL_NAME']
llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
embeddings = OllamaEmbeddings()
if vec_db_name == 'FAISS':
    vector_db = FAISS.load_local("faiss_index", embeddings)
if vec_db_name == 'CHROMA':
    vector_db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)
# if vec_db_name == 'QDRANT':
#     vector_db = Qdrant.load_local("qdrant_index", embeddings)
chain = load_qa_chain(llm, chain_type='stuff')
add_company_logo()


# Generate OpenAI Embeddings and indexing vector DB
def query_answer(query):
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response
        
user_lang =st.selectbox('Select Language', ('English', 'Tamil', 'Hindi'))

def chatbox(source, target):

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask question about PDF content?"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty() 
            raw_prompt = translate_text(prompt, target, source)
            result = translate_text(query_answer(raw_prompt), source, target) 
            result2 = ""
            for chunk in result.split():
                result2 += chunk + " "
                time.sleep(0.3)
                message_placeholder.markdown(result2 + "▌")

        st.session_state.messages.append({"role": "assistant", "content": result})

if user_lang == 'Tamil':
    chatbox('en','tamil')

if user_lang == 'English':
    chatbox('auto','en')

if user_lang == 'Hindi':
    chatbox('en','hi')