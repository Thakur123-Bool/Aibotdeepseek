import os
import logging
import requests
import re
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
from huggingface_hub import login
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Hugging Face login
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables")
login(HUGGINGFACE_TOKEN)

# Set up the Llama pipeline
llama_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B",
    max_new_tokens=50,
    temperature=0.3,
    top_k=10,
    pad_token_id=0
)

# Strict prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""{question}\n\nContext:\n{context}\n\nAnswer:"""
)

# Initialize FastAPI app
app = FastAPI()

# Global variables
retrieval_qa_chain = None

# Extract text from PDF
def extract_text_from_pdf(file: BytesIO):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text, None
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None, f"Error reading PDF: {e}"

# Split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# Create FAISS vectorstore
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

# Initialize RetrievalQA
def initialize_retrieval_qa(vectorstore):
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": custom_prompt}
    )

# Clean response and format output
def format_response(question, response):
    try:
        response = re.sub(r"(Context:.*|Answer:)", "", response, flags=re.IGNORECASE).strip()
        return f"<p>{response}</p>"
    except Exception as e:
        logging.error(f"Error formatting response: {e}")
        return response

# Process documents (files or URL)
def process_documents(uploaded_files: List[UploadFile] = None, url: str = None):
    global retrieval_qa_chain
    text = ""
    status = []

    if uploaded_files:  # Local file upload
        status.append("Processing uploaded files...")
        for file in uploaded_files:
            file_content = BytesIO(file.file.read())
            file_text, error = extract_text_from_pdf(file_content)
            if error:
                return f"{error}\nStatus: {status[-1]}"
            text += file_text
            status.append(f"Processed file: {file.filename}")
    elif url:  # URL download
        status.append(f"Downloading from URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return f"Error: Failed to download (Status {response.status_code})\nStatus: {status[-1]}"
            file_content = BytesIO(response.content)
            file_text, error = extract_text_from_pdf(file_content)
            if error:
                return f"{error}\nStatus: {status[-1]}"
            text = file_text
            status.append("Downloaded and processed PDF from URL")
        except Exception as e:
            return f"Error downloading from URL: {e}\nStatus: {status[-1]}"

    if not text.strip():
        return "Error: No text found in the documents.\nStatus: " + "\n".join(status)

    try:
        status.append("Splitting text into chunks...")
        text_chunks = split_text_into_chunks(text)
        status.append("Creating vectorstore...")
        vectorstore = create_vectorstore(text_chunks)
        status.append("Initializing QA chain...")
        retrieval_qa_chain = initialize_retrieval_qa(vectorstore)
        return "Documents processed successfully. Ask your questions!\nStatus: " + "\n".join(status)
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        return f"Error: {e}\nStatus: " + "\n".join(status)

# Answer questions
@app.post("/ask_question/")
async def ask_question(question: str):
    if retrieval_qa_chain is None:
        raise HTTPException(status_code=400, detail="Documents have not been processed yet.")
    try:
        response = retrieval_qa_chain.run({"query": question})
        formatted_response = format_response(question, response)
        return {"response": formatted_response}
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# Upload documents
@app.post("/upload_documents/")
async def upload_documents(uploaded_files: List[UploadFile] = File(...)):
    status = process_documents(uploaded_files=uploaded_files)
    return {"status": status}

# URL for documents
@app.post("/process_url/")
async def process_url(url: str):
    status = process_documents(url=url)
    return {"status": status}

# Default route
@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Question Answering API"}
