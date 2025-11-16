import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

## load the Groq API key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


st.title("PDFs Q&A App using Nvidia NIM and Langchain")
## Input the Nvidia NIM API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

if not api_key:
    st.warning("Please provide NVIDIA NIM Api Key")
    st.stop()


llm = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key=api_key
)


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
{context}
Questions:{input}

"""
)

user_prompt=st.text_input("Enter Your Question From Doduments")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")


if user_prompt:
    retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 2})
    chain=(
        {"context": retriever, "input": RunnablePassthrough()}
        |prompt
        |llm
        )
    start=time.process_time()
    response=chain.invoke(user_prompt)
    print("Response time :",time.process_time()-start)
    st.write(response.content)
