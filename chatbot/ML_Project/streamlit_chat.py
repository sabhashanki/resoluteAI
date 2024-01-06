import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import time
from deep_translator import GoogleTranslator


# Load environment variables
load_dotenv()


def translate_text(text, source='auto', target='ta'):
    return GoogleTranslator(source=source, target=target).translate(text)

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    st.write('Creating embeddings.... Please wait')
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    st.write('Embeddings generated... Now you can ask questions ?')

    return new_db

def query_answer(query):
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response

st.title("Shankesh AI - Chat with your PDF 💬")

pdf = st.sidebar.file_uploader('Upload your PDF Document', type='pdf')
user_lang = st.sidebar.selectbox(
    "Select the language to chat ?",
    ("Tamil", "Hindi", "English")
)
process_button = st.sidebar.button('Process')

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    vector_db = process_text(text)    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')


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
        st.markdown(translate_text(prompt))

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        result = translate_text(query_answer(prompt))
        result2 = ""
        for chunk in result.split():
            result2 += chunk + " "
            time.sleep(0.3)
            message_placeholder.markdown(result2 + "▌")

    st.session_state.messages.append({"role": "assistant", "content": result})


        
