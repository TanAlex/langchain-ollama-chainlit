import os
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import (
#     DirectoryLoader,
#     PyPDFLoader,
# )
# from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Clone
repo_path = os.path.join(ABS_PATH, "data/repo")
# repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

# Create vector database
def create_vector_database():
    """
    Creates a vector database using GenericLoaders and embeddings.

    """
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    # len(documents)
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    chunked_documents = python_splitter.split_documents(documents)
    # help me print formated len(chunked_documents) 
    print(f"total documents chunks: {len(chunked_documents)}")


    # Initialize Ollama Embeddings
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()
  
    # query it
    #query = "Who are the authors of the paper"
    #docs = vector_database.similarity_search(query)


    # print results
    #print(docs[0].page_content)


if __name__ == "__main__":
    create_vector_database()
