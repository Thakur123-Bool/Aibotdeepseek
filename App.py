import os
import logging
import requests
import re
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up the DeepSeek API endpoint and authentication token (if needed)
DEEPSEEK_API_URL = os.getenv("https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("sk-053e67b2f237425ab79299424beff735")

# Initialize FastAPI app
app = FastAPI()


# Mount the static directory to serve HTML, CSS, and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable for storing DeepSeek embeddings and responses
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

# Split text into chunks (optional, DeepSeek might handle this internally)
def split_text_into_chunks(text):
    # Assuming DeepSeek handles chunking, if needed, adjust accordingly.
    return text.split("\n\n")

# Call DeepSeek API to generate embeddings or to process questions
def deepseek_query(query: str, documents: str):
    # You might need to adjust this to fit the exact API endpoint for DeepSeek
    response = requests.post(
        DEEPSEEK_API_URL + "/query", 
        json={"query": query, "documents": documents},
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error querying DeepSeek API: {response.text}")
    return response.json()

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
        status.append("Sending documents to DeepSeek API for processing...")
        # Assuming DeepSeek processes the entire document at once or after chunking
        response = deepseek_query("Extract information", text)
        if 'error' in response:
            return f"Error: {response['error']}\nStatus: " + "\n".join(status)
        status.append("DeepSeek API processed documents successfully.")
        return "Documents processed successfully. Ask your questions!\nStatus: " + "\n".join(status)
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        return f"Error: {e}\nStatus: " + "\n".join(status)

# Answer questions
@app.post("/ask_question/")
async def ask_question(question: str):
    if not retrieval_qa_chain:
        raise HTTPException(status_code=400, detail="Documents have not been processed yet.")
    try:
        # Make a query to DeepSeek API to get the answer
        response = deepseek_query(question, retrieval_qa_chain)
        return {"response": response['answer']}
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
    return {"message": "Welcome to the DeepSeek-powered PDF Question Answering API"}

# New endpoint to upload a file and description using Form data
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), description: str = Form(...)):
    return {"filename": file.filename, "description": description}
