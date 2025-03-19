import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Set Streamlit Page Configuration
st.set_page_config(page_title="RAG AI Assistant")

st.title("Personalized AI Assistant")

# Function to load the OpenAI LLM
def load_LLM(api_key):
    return ChatOpenAI(
        temperature=0.9, 
        openai_api_key=api_key, 
        model="gpt-3.5-turbo-0125", 
        streaming=True
    )

# Input field for OpenAI API Key
st.markdown("### Enter your OpenAI API Key")
def get_openai_api_key():
    return st.text_input("OpenAI API Key", placeholder="sk-...", type="password")

openai_api_key = get_openai_api_key()

# Upload PDF file
st.markdown("### Upload the PDF file you want to chat with")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file is not None:
    # Read PDF file without encoding issues
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open("pdf", pdf_bytes)

    # Extract text from the PDF
    pdf_text = "\n".join([page.get_text("text") for page in pdf_document])

    if len(pdf_text.split(" ")) > 20000:
        st.error("Please upload a shorter file. The maximum length is 20,000 words.")
        st.stop()

    if pdf_text and openai_api_key:
        # Convert extracted text into Document object
        documents = [Document(page_content=pdf_text)]

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        chunks_of_text = text_splitter.split_documents(documents)

        # Create vector database
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_db = Chroma.from_documents(chunks_of_text, embeddings)
        retriever = vector_db.as_retriever()

        # User input field for questions
        question = st.text_input("Ask a question about the document:")

        if question:
            # Retrieve relevant document chunks
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Generate prompt for LLM
            prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"

            # Load LLM
            llm = load_LLM(openai_api_key)
            st.markdown("### AI Response:")
            response_area = st.empty()  # Placeholder for response
            full_response = ""

            # Stream response in real-time
            for chunk in llm.stream(prompt):
                full_response += chunk
                response_area.markdown(full_response)  # Live update response
