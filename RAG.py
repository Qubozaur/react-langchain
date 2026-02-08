import os
from pydoc import doc
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.db_data.dataclasses import search_rerank

load_dotenv()

if __name__ == "__main__":
    Flag = 0
    if Flag:
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
    
    print("Initializing components ...")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context: 
        {context}
        
        Question: {question}

        Provide a detailed answer:"""
    )

    def format_docs(docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieval_chain_without_lcel(query: str):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        messages = prompt_template.format_messages(context= context, question=query)
        response = llm.invoke(messages)
        return response.content

    query = "What is Pinecone in machine learning ?"

    # no RAG
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content)

    print("="*70)

    # no lcel
    result_without_lcel = retrieval_chain_without_lcel(query=query)
    print(result_without_lcel)



