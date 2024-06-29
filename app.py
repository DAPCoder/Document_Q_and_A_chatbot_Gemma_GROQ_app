import streamlit as st                        # Build quick and easy applications
import os
from langchain_groq import ChatGroq           # Chatbot implementation from GROQ
from langchain.text_splitter import RecursiveCharacterTextSplitter            #Split texts into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain   # Get the relevant documents to help set up context
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS                            # for vector store database/ store vectors for Symantec /similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader         # to read pdf files from directory
from langchain_google_genai import GoogleGenerativeAIEmbeddings               # for embedding / converts chunks of texts into embeddings
from dotenv import load_dotenv

# load environment variables
load_dotenv()

## load the GROQ And OpenAI API KEY
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# call GROQ model
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

# Test : View in terminal
# print(llm)

# Set up Prompt template
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# Create Vector embedding function
# Read all documents from pdf files and
# convert into chunks and apply Google Generative AI embeddings
# Store embeddings in vector store db (FAISS) by Facebook
# Keep Vector store DB in session state


def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")                  #embedding model selection
        st.session_state.loader=PyPDFDirectoryLoader("./robotics_data")                   ## Data Ingestion -> can switch to "us_census_data"                    
        st.session_state.docs=st.session_state.loader.load()                          ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)            ## Chunk Creation  /chunking documents -> breaking into sets
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # document splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) # FIASS vector database will store embeddings /vector OpenAI embeddings





prompt1=st.text_input("Ask your question based on the documents")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
