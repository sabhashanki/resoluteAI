from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from utils import translate_text, add_company_logo
import time

# Initialization
llm = OpenAI()
embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local("faiss_index", embeddings)
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
            st.markdown(translate_text(prompt, source, target))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty() 
            result = translate_text(query_answer(prompt), source, target) 
            result2 = ""
            for chunk in result.split():
                result2 += chunk + " "
                time.sleep(0.3)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(result2 + "▌")
            # message_placeholder.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

if user_lang == 'Tamil':
    chatbox('auto','tamil')

if user_lang == 'English':
    chatbox('auto','en')

if user_lang == 'Hindi':
    chatbox('auto','hi')