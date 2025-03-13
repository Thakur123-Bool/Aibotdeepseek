import os
import logging
import requests
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up the DeepSeek API endpoint and authentication token (ensure correct variable names)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")  # Correct the variable name
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Correct the variable name

# Initialize FastAPI app
app = FastAPI()

# Global variable for storing processed text for DeepSeek query
retrieval_qa_chain = None

# Extract text from PDF file
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

# DeepSeek query function
def deepseek_query(query: str, documents: str):
    response = requests.post(
        DEEPSEEK_API_URL + "/query", 
        json={"query": query, "documents": documents},
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error querying DeepSeek API: {response.text}")
    return response.json()

# Process documents (either from file or URL)
def process_documents(uploaded_files: List[UploadFile] = None, url: str = None):
    global retrieval_qa_chain
    text = ""
    status = []

    # If uploaded files are provided
    if uploaded_files: 
        status.append("Processing uploaded files...")
        for file in uploaded_files:
            file_content = BytesIO(file.file.read())
            file_text, error = extract_text_from_pdf(file_content)
            if error:
                return f"{error}\nStatus: {status[-1]}"
            text += file_text
            status.append(f"Processed file: {file.filename}")
    
    if not text.strip():
        return "Error: No text found in the documents.\nStatus: " + "\n".join(status)

    # Store the processed text for querying
    retrieval_qa_chain = text
    status.append("Documents processed successfully.")
    return "Documents processed successfully. Ask your questions!\nStatus: " + "\n".join(status)

# API endpoint to answer questions using DeepSeek
@app.post("/ask_question/")
async def ask_question(question: str):
    if not retrieval_qa_chain:
        raise HTTPException(status_code=400, detail="Documents have not been processed yet.")
    try:
        response = deepseek_query(question, retrieval_qa_chain)
        return {"response": response.get('answer', 'No answer available')}
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# API endpoint to upload documents
@app.post("/upload_documents/")
async def upload_documents(uploaded_files: List[UploadFile] = File(...)):
    status = process_documents(uploaded_files=uploaded_files)
    return {"status": status}

# API endpoint to upload file with description (description is optional)
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), description: str = Form(...)):
    return {"filename": file.filename, "description": description}

# Default root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the DeepSeek-powered PDF Question Answering API"}
