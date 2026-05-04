import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, DOC_SOURCE_DIR

pinecone = Pinecone(api_key=PINECONE_API_KEY)


# Create index if doesn't exist
def create_index():
    if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Index created")
    else:
        print("Index already exists")
    return pinecone.Index(PINECONE_INDEX_NAME)
    