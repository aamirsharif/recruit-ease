# import faiss
# import pandas as pd
# import os
# import pickle
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores.faiss import DistanceStrategy
# from langchain_community.document_loaders import DataFrameLoader

# # Define paths
# DATA_PATH = "../data/main-data/synthetic-resumes1.csv"
# FAISS_PATH = "../vectorstore"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# def ingest(df: pd.DataFrame, content_column: str, embedding_model, index_filename: str = "index.faiss", metadata_filename: str = "index.pkl"):
#     # Ensure the directory exists
#     os.makedirs(FAISS_PATH, exist_ok=True)

#     # Initialize the DataFrameLoader
#     loader = DataFrameLoader(df, page_content_column=content_column)

#     # Initialize the text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1024,
#         chunk_overlap=500
#     )

#     # Load the documents
#     documents = loader.load()
    
#     # Split documents into chunks
#     document_chunks = text_splitter.split_documents(documents)

#     # Create the FAISS vector store
#     vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)

#     # Save the FAISS index
#     faiss.write_index(vectorstore_db.index, os.path.join(FAISS_PATH, index_filename))

#     # Collect metadata (e.g., document IDs)
#     # Example: metadata is assumed to be a list of document IDs or any relevant information
#     metadata = [doc.metadata for doc in document_chunks]

#     # Save the metadata as a pickle file
#     with open(os.path.join(FAISS_PATH, metadata_filename), "wb") as f:
#         pickle.dump(metadata, f)

#     return vectorstore_db




import faiss
import pandas as pd
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader

# Define paths
DATA_PATH = "../data/main-data/resumes1.csv"
FAISS_PATH = "../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest(df: pd.DataFrame, content_column: str, embedding_model, index_filename: str = "index.faiss", metadata_filename: str = "index.pkl"):
    # Ensure the directory exists
    os.makedirs(FAISS_PATH, exist_ok=True)

    # Initialize the DataFrameLoader
    loader = DataFrameLoader(df, page_content_column=content_column)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=500
    )

    # Load the documents
    documents = loader.load()
    
    # Split documents into chunks
    document_chunks = text_splitter.split_documents(documents)

    # Create the FAISS vector store
    vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)

    # Save the FAISS index
    faiss.write_index(vectorstore_db.index, os.path.join(FAISS_PATH, index_filename))

    # Collect metadata (e.g., document IDs)
    metadata = {
        'docstore': vectorstore_db.docstore,
        'index_to_docstore_id': vectorstore_db.index_to_docstore_id
    }

    # Save the metadata as a pickle file
    with open(os.path.join(FAISS_PATH, metadata_filename), "wb") as f:
        pickle.dump(metadata, f)

    return vectorstore_db
