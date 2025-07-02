import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import os

os.environ["GROQ_API_KEY"] = "my_key"

st.header('Magnusbot')

#Upload the pdf
with st.sidebar:
    st.title("Your Document")
    file=st.file_uploader("Upload you pdf", type="pdf")
    
#Extract the text
if file is not None:
    pdf_pages=PdfReader(file)
    text=""
    for page in pdf_pages.pages:
        text+=page.extract_text()
        # st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    # st.write(chunks)
    
#Generate Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    
#Create vector store
    vector_store = FAISS.from_texts(chunks, hf)
    
#Get a user query
    user_query=st.text_input("Enter your query")
    
    if user_query:
        match = vector_store.similarity_search(user_query)
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_retries=2,
        )
        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=match,question=user_query)
        
        st.subheader("Answer")
        st.write(response)