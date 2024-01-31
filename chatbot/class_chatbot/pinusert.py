from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAI
from dotenv import load_dotenv
import pinecone
load_dotenv()

pinecone.init(environment='gcp-starter')
llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
chain = load_qa_chain(llm, chain_type='stuff')
index_name = "langchainvector"
embeddings = HuggingFaceEmbeddings(
    model_name="./paraphrase-MiniLM-L6-v2/",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
index = pinecone.Index(index_name)


def upsertData(model, chunks):
    upserted_data = []
    i = 0
    for item in chunks:
        id = index.describe_index_stats()['total_vector_count']
        upserted_data.append(
            (
                str(id+i),
                model.encode(item).tolist(),
                {
                    'content' : item
                }
            )
        )
        i += 1
    index.upsert(vectors=upserted_data)


rawText = """
SHANKESH RAJU MS
Chennai, India | +919444941332 | shankeshraju@gmail.com | https://github.com/sabhashanki | https://linkedin.com/in/shankeshrajums | https://huggingface.co/sabhashanki
EXPERIENCE
ResoluteAI Software
Data Science Intern - Generative AI (December 2023 – Till Present)
Roles and Responsibilities
• Designing, developing, and implementing generative AI models and algorithms utilising state-
of-the-art techniques such as Generative Pre-Trained Transformers (GPT), Variational auto
encoder, and Generative adversarial networks (GAN)
• Research to stay up-to-date with the latest advancements in generative AI, machine learning,
and deep learning techniques and identify opportunities to integrate them into our products
and services.
• Optimising existing generative AI models for improved performance, scalability, and efficiency
• Developing and maintaining AI pipelines, including data preprocessing, feature extraction,
model training, and evaluation
"""

textSpliter = CharacterTextSplitter(
    separator='\n',
    chunk_size=800,
    chunk_overlap=50,
    length_function=len
)
chunks = textSpliter.split_text(rawText)
# model = SentenceTransformer('./paraphrase-MiniLM-L6-v2/')
test = embeddings.embed_query('name of the candidate')

# print(test[0].shape)
vectorDB = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
# upsertData(model, chunks)
query = 'What is the name of the candidate and mail address ?'
# queryEmbed = model.encode(query).tolist()
# response = index.query(queryEmbed, top_k=2, includeMetadata=True)
# vectorDB = Pinecone(index, embeddings.embed_query, query)
docs = vectorDB.similarity_search(query)
response = chain.run(input_documents=docs, question=query)
print(response)
# print(response['matches'][0]['metadata']['content'])




