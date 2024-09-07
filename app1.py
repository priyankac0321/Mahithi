import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an advanced AI system designed to provide detailed information about loan options using the information stored in multiple text documents. Your goal is to assist users by answering their questions about various loan options available, drawing upon the context and data provided in these documents.

When asked about loan options in a specific domain, respond with detailed information including:

1. Types of loans available in the specified domain (e.g., personal loans, mortgage loans, student loans, auto loans).
2. Key features of each loan type (e.g., interest rates, repayment terms, eligibility criteria).
3. Advantages and disadvantages of each loan type.
4. Specific requirements or considerations for each loan type.

If the requested information is not available in the provided context, respond with "The information is not available in the context." Do not provide incorrect or speculative answers.

Remember, your purpose is to inform users about their loan options as comprehensively and accurately as possible, using the textual data provided.

    Context:\n {context}?\n
    Domain: {domain}\n

    List all loan options available in the given domain with detailed information.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "domain"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "domain": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def get_text_from_files(file_paths):
    text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text += file.read() + "\n"
    return text

def main():
    st.set_page_config(page_title="Mahithi")
    st.header("Mahithi: Cognitivi Assistant")

    # Predefined directory containing text files
    text_files_directory = './data'
    
    # List all text files in the directory
    text_file_paths = [os.path.join(text_files_directory, file) for file in os.listdir(text_files_directory) if file.endswith('.txt')]

    if not text_file_paths:
        st.error("No text files found in the specified directory.")
        return

    # Process files and create vector store
    with st.spinner("Processing files..."):
        raw_text = get_text_from_files(text_file_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    st.success("File processing completed")

    # Allow user to ask questions
    user_question = st.text_input("Ask a Question from the Text Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
