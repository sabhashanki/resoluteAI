from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone, FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from deep_translator import GoogleTranslator
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
import configparser
import pinecone
import os

# Initialization
load_dotenv()
config = configparser.ConfigParser()
index_name = "langchainvector"
config.read('./config.ini')
vec_db_name = config['VECTOR_DB']['MODEL_NAME']
embed_model = config['EMBEDDING_MODEL']['MODEL_NAME']
model_path = config['EMBEDDING_MODEL']['MODEL_REPO']
data_folder = Path(config['INPUT']['DATA_FOLDER'])
output_folder_text = Path(config['OUTPUT']['TEXT_FOLDER'])
output_folder_embeddings = Path(config['OUTPUT']['EMBEDDINGS_FOLDER'])
output_language = str(config['LANGUAGE']['LANG'])



# Embedding model selection
def initFunc():
    print('Initialization inside function')
    pinecone.init(environment="gcp-starter")
    llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
    if embed_model == 'OPENAI':
        embeddings = OpenAIEmbeddings()
    if embed_model == 'HUGGINGFACE':
        embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
    return embeddings, llm



# Extract text from document
def extract_text():
    pdf_pages  = 0
    text = ""
    if (data_folder.exists() and data_folder.is_dir()):
        for data_file in data_folder.iterdir():
            if data_file.is_file() and data_file.suffix.lower()==".pdf":
                pdfdata = PdfReader(data_file)
                for page in pdfdata.pages:
                    pdf_pages += 1
                    text += page.extract_text()
            if data_file.is_file() and data_file.suffix.lower() in ['.txt', '.rst', '.md']:
                print(f'file type is {data_file.suffix.lower()}')
                with open(data_file, 'r') as file:
                    for line in file:
                        text += line
        if not output_folder_text.exists:
            os.mkdir(output_folder_text)
        output_path = os.path.join(output_folder_text, 'raw_text.txt')
        file = open(output_path, 'w')
        file.write(text)
        print(f'Extraced {pdf_pages} pages and saved in {output_path}')
    else:
        print(f"{data_folder} is not a valid input directory")
    return text



# Raw text into chunks
def process_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    print(f'text split into chunks : {len(chunks)}')
    return chunks



# Translate
def translate_text(text, source='auto', target='ta'):
    return GoogleTranslator(source=source, target=target).translate(text)



# Creating Embeddngs and VectorStore
def embedVectorStore(embeddings, chunks):
    try:
        if vec_db_name == 'PINECONE':
            print(f'Creating {embed_model} embeddings with {vec_db_name}.... Please wait')
            vector_db = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
            print('Embeddings created and saved in the pinecone vector database')
        if vec_db_name == 'FAISS':
            print(f'Creating {embed_model} embeddings with {vec_db_name}.... Please wait')
            vector_db = FAISS.from_texts(chunks, embeddings)
            vector_db.save_local(f"{output_folder_embeddings}/faiss_index")
            print('Embeddings created and saved in the FAISS vector store')
    except Exception as e:
        print(f'Embedding creation failed: {e}')
    return vector_db



def main():
    embeddings, llm = initFunc()
    raw_text = extract_text()
    chunks = process_text(raw_text)
    vector_db = embedVectorStore(embeddings, chunks)
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff', 
        retriever=vector_db.as_retriever(),
        )
    while True:
        text_input = input("\nUser: ")
        if text_input == "exit":
            break
        response = chain.invoke(text_input)['result']
        final_response = translate_text(response , 'auto', output_language)
        print(f"Agent: {final_response}")



if __name__ == "__main__":
    main()