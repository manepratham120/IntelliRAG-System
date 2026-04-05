import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

embedding = HuggingFaceEmbeddings()

# llm
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

def process_document_to_chroma_db(filename):
    loader = PyPDFLoader(f"{working_dir}/{filename}")
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )
    
    texts = text_splitter.split_documents(document)
    
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    
    return 0

def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    
    retriever = vectordb.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]
    
    return answer