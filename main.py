import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="RAG AI Assistant",
)

st.title("Personalized AI Assistant")

def load_LLM():
    # Ensure your OpenAI API key is set as an environmental variable or entered by the user
    return ChatOpenAI(
        temperature=0.9, 
        openai_api_key=openai_api_key, 
        model="gpt-3.5-turbo-0125", 
        streaming=True
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown("Welcome to my first RAG app!")

with col2:
    st.write(
        "This app is powered by OpenAI's GPT-3.5-turbo model. Visit [openai.com](https://openai.com) for more information."
    )

# Input OpenAI API Key
st.markdown("### Enter your OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key",
        placeholder="Ex: sk-2twmA8tfCb8un4...",
        key="openai_api_key_input",
        type="password",
    )
    return input_text

openai_api_key = get_openai_api_key()

# Input PDF
st.markdown("### Upload the PDF file you want to chat with")

uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file is not None:
    # Open the uploaded PDF file using PyMuPDF
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # Extract text from the PDF document
    pdf_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text("text")

    if len(pdf_text.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20,000 words.")
        st.stop()

    if pdf_text:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key.", icon="⚠️")
            st.stop()

        # Convert the extracted text into a list of Document objects
        documents = [Document(page_content=pdf_text)]  # Wrap the text into a Document object

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=500,
            chunk_overlap=250
        )

        # Split the document into chunks using the Document object
        chunks_of_text = text_splitter.split_documents(documents)

        # Initialize Chroma without using deprecated client settings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_db = Chroma.from_documents(chunks_of_text, embeddings)
        retriever = vector_db.as_retriever()

        # Add a text input for the user's question
        question = st.text_input("Ask a question about the document:")

        if question:
            # Use the retriever to get relevant document chunks for the question
            relevant_docs = retriever.invoke(question)

            # Prepare the context from the relevant documents
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Concatenate the question and context into a single string
            prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"

            # Call the LLM with streaming
            llm = load_LLM()
            st.markdown("### AI Response:")
            response_area = st.empty()  # Create an empty container for the response
            full_response = ""

            for chunk in llm.stream(prompt):  # Use stream instead of invoke
                full_response += chunk.content  # Accumulate the chunks
                response_area.markdown(full_response)  # Update the response in real time
