import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, DOC_SOURCE_DIR

pinecone = Pinecone(api_key=PINECONE_API_KEY)

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['PINECONE_ENVIRONMENT'] = PINECONE_ENVIRONMENT

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

INDEX_NAME = PINECONE_INDEX_NAME

def get_retriever():
    """Initilaizes and reitrives the pincone reitriver """

    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index ")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created")
    

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()

  
def add_document(text_content: str):
    """Add a single doucment in the pinecone index database"""
    if not text_content:
        raise ValueError("Text content cannot be empty")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


    documents = text_splitter.create_documents([text_content])
    print("splitting the document")


    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    vectorstore.add_documents(documents)
    print("document added sucess")