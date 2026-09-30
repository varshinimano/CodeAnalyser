from src.helper import load_repo, text_splitter, load_embedding

from dotenv import load_dotenv

from langchain_chroma import Chroma

import os


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


documents = load_repo("repo/")

print("Number of documents:", len(documents))


text_chunks = text_splitter(documents)

print("Number of chunks:", len(text_chunks))


embeddings = load_embedding()


# Store vectors in ChromaDB

vectordb = Chroma.from_documents(
    text_chunks,
    embedding=embeddings,
    persist_directory="./db"
)


print("Vector database created successfully.")