import os
from pydoc import doc
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

if __name__ == "__main__":
    print("Loading data ...")
    loader = TextLoader("mediumblogRAG.txt", encoding="UTF-8")
    document = loader.load()
    print("Data loaded!\n")

    print("Splitting into chunks ...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=8)
    texts = text_splitter.split_documents(documents=document)
    print(f"Created: {len(texts)} chunks\n")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting ...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("Finished!")



