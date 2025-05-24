import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables or .env file.")
lm=ChatGroq(GROQ_API_KEY=GROQ_API_KEY,model_name="llama3-8b-8192")
prompt=ChatPromptTemplate.from_template(
    """Answer the questions based on m the provided context only.
       Please provide the most accurate response based on the question.
       <context>
       {context}
       </context>
       question:{question}
"""
)
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
user_prompt=st.text_input("enter your query from research paper")
if st.button("Document embeddings"):
    create_vector_embeddings()
    st.write("vector database is ready")
import time
if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()

    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    
    start=time.process_time()
    response=retrieval_chain.invoke({"input":user_prompt})
    print("response time",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['content']):
            st.write(doc.page_content)
            st.write('***********')

