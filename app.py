import streamlit as st
st.set_page_config(page_title="Chat App", page_icon="💬", layout="centered")
# Add custom CSS to hide the GitHub icon
# st.markdown(
#     """
#     <style>
#     [data-testid="stToolbar"]{
#         display: none !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
import os
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import PyPDF2
import faiss
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO  
from together import Together
import re
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import subprocess
import logging
import sys
import asyncio
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions
import openai
import hashlib
from hashlib import md5
import random
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
# ================== Environment Variables ==================
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
st.write("TOGETHER_API_KEY:", TOGETHER_API_KEY)
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY is not loaded. Check your .env file!")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =================== Connections ============================
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
users_collection = db["users"]

client = Together(api_key=TOGETHER_API_KEY)
# =================== Authentication UI =========================

# Hash the password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if a user exists
def get_user(username):
    return users_collection.find_one({"username": username})

# Create a new user in MongoDB
def create_user(username, password, email):
    hashed_pw = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed_pw, "email": email})

# Authenticate user credentials
def authenticate_user(username, password):
    user = get_user(username)
    return user and user["password"] == hash_password(password)

# Update the password for a user
def update_password(username, new_password):
    hashed_pw = hash_password(new_password)
    users_collection.update_one({"username": username}, {"$set": {"password": hashed_pw}})

# Generate a 6-digit OTP
def generate_otp():
    return str(random.randint(100000, 999999))

# Send OTP to user email
def send_otp_email(email, otp):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Password Reset OTP"
        msg["From"] = "nancyhisham2003@gmail.com"
        msg["To"] = email
        msg.set_content(f"Your OTP for password reset is: {otp}")

        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("nancyhisham2003@gmail.com", "sike xztt teak orkr")
            server.send_message(msg)
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Authentication UI
with st.sidebar:
    if not st.session_state["authenticated"]:
        st.header("🔐 Authentication")
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

        # **🔹 Login Tab**
        with tab1:
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

        # **🔹 Sign-Up Tab**
        with tab2:
            st.subheader("Create an Account")
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Enter Your Email")

            if st.button("Sign Up"):
                try:
                    validate_email(email)
                    if new_password != confirm_password:
                        st.error("❌ Passwords do not match.")
                    elif get_user(new_username):
                        st.error("❌ Username already exists. Try another.")
                    else:
                        create_user(new_username, new_password, email)
                        st.success("✅ Account created successfully! You can now log in.")
                except EmailNotValidError:
                    st.error("❌ Invalid email format.")

        # **🔹 Forgot Password Tab**
        with tab3:
            st.subheader("Reset Password")
            reset_username = st.text_input("Enter your Username")

            if st.button("Verify Username"):
                user = get_user(reset_username)
                if user:
                    otp = generate_otp()
                    st.session_state["reset_user"] = reset_username
                    st.session_state["otp"] = otp
                    st.session_state["email"] = user["email"]

                    if send_otp_email(user["email"], otp):
                        st.success("✅ OTP sent to your email! Enter it below.")
                    else:
                        st.error("❌ Failed to send OTP. Check email settings.")
                else:
                    st.error("❌ Username not found.")

            if "reset_user" in st.session_state:
                entered_otp = st.text_input("Enter OTP")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")

                if st.button("Reset Password"):
                    if entered_otp == st.session_state.get("otp"):
                        if new_password == confirm_new_password:
                            update_password(st.session_state["reset_user"], new_password)
                            st.success("✅ Password reset successfully! You can now log in.")
                            del st.session_state["reset_user"]
                            del st.session_state["otp"]
                            del st.session_state["email"]
                        else:
                            st.error("❌ Passwords do not match.")
                    else:
                        st.error("❌ Incorrect OTP.")

# **STOP rendering content if user is not authenticated**
if not st.session_state["authenticated"]:
    st.warning("⚠️ Please log in to access the application.")
    st.stop()

# =================== Main App =========================


def initialize_vector_db():
    model_local = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    index = faiss.IndexFlatL2(768)
    return model_local, index

embedding_model, faiss_index = initialize_vector_db()

AVAILABLE_MODELS_DICT = {
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini"},
    "openai-4o": {"price": "Custom", "type": "openai"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together"},
    "databricks/dbrx-instruct": {"price": "$1.20", "type": "together"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together"},
}

AVAILABLE_MODELS = list(AVAILABLE_MODELS_DICT.keys())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.",
        "selected_models": AVAILABLE_MODELS[:1],
        "vary_temperature": 1,
        "vary_top_p": 0
    }

def save_config(config):
    """Save configuration as a JSON file."""
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

def load_config(uploaded_file):
    """Load configuration from a JSON file uploaded by the user."""
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")

# -----------------------------------------------------------------------------
# PDF Processing Functions
# -----------------------------------------------------------------------------
# Function to chunk text with overlapping windows
def chunk_text(text, chunk_size=750, min_chunk_length=50, overlap_size=100):
    paragraphs = re.split(r'\n{2,}', text)  # Split by double newlines (paragraphs)
    chunks = []
    
    temp_chunk = ""
    prev_chunk = ""  # Store the last chunk for overlap

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)  # Split into sentences

        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size:
                temp_chunk += sentence + " "
            else:
                if len(temp_chunk.strip()) >= min_chunk_length:
                    chunks.append(prev_chunk[-overlap_size:] + temp_chunk.strip())  # Add overlap
                prev_chunk = temp_chunk.strip()  # Store the last chunk
                temp_chunk = sentence + " "

        # Ensure last part of paragraph is added
        if len(temp_chunk.strip()) >= min_chunk_length:
            chunks.append(prev_chunk[-overlap_size:] + temp_chunk.strip())  # Add overlap

    return chunks


def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text
# ================== Generate Response ==================

def update_vector_db(username, summary, texts, filehash, filename="uploaded"):
    if not texts:
        return
    embeddings = embedding_model.encode(texts).tolist()
    summary_embedding = embedding_model.encode([summary]).tolist()[0]
    # embedding_array = np.array(embeddings, dtype="float32")
    # print(f"Embedding shape: {embedding_array.shape}")
    documents = [{"username": username, "filename": filename, "text": text, "summary": summary, "filehash": filehash, "embedding": emb, "summary_embedding": summary_embedding} for text, emb in zip(texts, embeddings)] 
    try:
        users_collection.insert_many(documents, ordered=False)
    except Exception as e:
        print(f"Error inserting documents: {e}")

    faiss_index.add(np.array(embeddings, dtype="float32"))

def chunk_text_for_summary(text, chunk_size=3000):  # Reduce chunk size to ~2,000 words (~3,000 tokens)
    """Splits text into smaller chunks to fit GPT-4's token limit."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_chunk(chunk):
    """Summarizes a single chunk using OpenAI API."""
    response = openai.chat.completions.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" if you want it to be cheaper and slightly faster
        messages=[
            {"role": "system", "content": "Summarize the following text concisely:"},
            {"role": "user", "content": chunk}
        ]
    )
    return response.choices[0].message.content.strip()

def summarize_pdf(pdf_text):
    """Summarizes long PDFs by breaking them into smaller chunks."""
    text_chunks = chunk_text_for_summary(pdf_text)
    
    summaries = []
    for i, chunk in enumerate(text_chunks):
        summary = summarize_chunk(chunk)
        summaries.append(summary)

    return " ".join(summaries)

def compress_summary(summary_text, max_words=100):

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Summarize the following text in {max_words} words or less, keeping key insights only:"},
            {"role": "user", "content": summary_text}
        ]
    )

    return response.choices[0].message.content.strip()

def process_pdf(username, file, filehash=None, filename="uploaded"):
    text = extract_text(file)
    summary = summarize_pdf(text)
    if len(summary.split()) > 200:
        summary = compress_summary(summary, max_words=100)
    chunks = chunk_text(text)
    update_vector_db(username, summary, chunks, filehash, filename)
    return summary, chunks

# -----------------------------------------------------------------------------
# AI Generation Functions
# -----------------------------------------------------------------------------
def generate_response_gemini(prompt, context, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    input_parts = [system_prompt + "\n" + context, prompt]
    generation_config = genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=temp,
        top_p=top_p,
        top_k=32
    )
    retries = 3
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(input_parts, generation_config=generation_config)
            return response.text
        except exceptions.ResourceExhausted:
            st.warning(f"API quota exceeded. Retrying... ({attempt+1}/{retries})")
            time.sleep(5)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    st.error("API quota exceeded. Please try again later.")
    return "Error generating response."

def generate_response_openAi(prompt, context, temp, top_p):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response(prompt, context, model, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"Context: {context}. Question: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def evaluate_summaries_with_gpt(question, summaries):
    relevant_summaries = []

    for summary in summaries:
        messages = [
            {"role": "system", "content": "Determine if the following summary is relevant to answering the user's question. Reply only with 'Relevant' or 'Not Relevant'."},
            {"role": "user", "content": f"User Question: {question}\n\nSummary: {summary}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        decision = response.choices[0].message.content.strip()
        if decision == "Relevant":
            relevant_summaries.append(summary)

    return relevant_summaries
# -----------------------------------------------------------------------------
# Retrieval Function (RAG)
# -----------------------------------------------------------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_context(username, query, top_summary_k=5, top_chunk_k=15):
    if not username:
        return []
    
    # Step 1: Get the query embedding
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # Step 2: Retrieve all stored summaries and their embeddings
    stored_summaries = list(users_collection.find({"username": username}, {"_id": 0, "summary_embedding": 1, "summary": 1}))
    
    if not stored_summaries:
        return []
    
    # Step 3: Extract summary embeddings and texts
    summary_embeddings = np.array([doc["summary_embedding"] for doc in stored_summaries if "summary_embedding" in doc], dtype="float32")
    summaries = [doc["summary"] for doc in stored_summaries if "summary" in doc]


    # Step 4: Find the most relevant summaries
    similarities = cosine_similarity([query_embedding], summary_embeddings)[0]
    top_summary_indices = np.argsort(similarities)[-top_summary_k:][::-1]
    relevant_summaries = [summaries[i] for i in top_summary_indices]

    relevant_summaries_by_gpt = evaluate_summaries_with_gpt(query, relevant_summaries)

    # Step 5: Retrieve chunks from the selected summaries
    stored_chunks = list(users_collection.find({"username": username}, {"_id": 0, "embedding": 1, "text": 1, "summary": 1}))
    relevant_chunks = []
    
    for summary in relevant_summaries_by_gpt:
        chunks = [doc for doc in stored_chunks if doc.get("summary") == summary]
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
        if faiss_index.ntotal == 0:
            faiss_index.add(np.array(chunk_embeddings, dtype="float32"))

        # Step 6: Find the most relevant chunks within these summaries
        if len(chunk_texts) > 0:
            distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), min(top_chunk_k, len(chunk_texts)))
            for i in indices[0]:
                if i < len(chunk_texts):
                    relevant_chunks.append(chunk_texts[i])

    return relevant_chunks



# ========= PDFs Link Extraction via URL =========
def get_page_items(url, base_url, listing_endpoint):
    """Extracts all item links from a page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = set()

        for item in soup.find_all("a"):
            link = item.get("href")
            # title = item.text.strip()
            if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"):
                    link = base_url + link
                items.add(link)
            

        return list(items)
    
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []


def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
    """Scrapes multiple pages to collect all links."""
    all_items = set()

    for page in range(1, num_pages + 1):
        url = f"{base_url}/{listing_endpoint}/{pagination_format}{page}"
        items = get_page_items(url, base_url, listing_endpoint)
        if not items:
            break
        
        all_items.update(items)
        st.write(f"Scraped page {page}")

    return list(all_items)

# =================== Try another way ============================
async def fetch_page(url):
    """Fetch page content asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=None)
        return response.text, str(response.url)


async def download_with_retries(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()
                return response.content
        except (httpx.ReadTimeout, httpx.RequestError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise Exception(f"Failed to download {url} after {retries} attempts")

async def extract_info(url):
    """Extract text and PDF links from a webpage."""
    html, base_url = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    pdf_links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]

    return pdf_links

async def main(urls):
    """Scrape multiple pages concurrently."""
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
def delete_file(username, filename):
    users_collection.delete_many({"username": username, "filename": filename})
    st.rerun()

def delete_all_files(username):
    users_collection.delete_many({"username": username})
    st.rerun()

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
async def store_in_DB(username, pdf_links):
    async with aiohttp.ClientSession() as session:
        unique_file_hashes = set(
            item.get("filehash") for item in users_collection.find({"username": username}, {"filehash": 1})
            if "filehash" in item
        )

        for pdf_link in pdf_links:
            try:
                async with session.get(pdf_link) as response:
                    if response.status == 200:
                        with st.spinner(f"Processing PDF: {pdf_link}"):
                            pdf_bytes = await response.read()
                            filehash = hashlib.md5(pdf_bytes).hexdigest()
                            if filehash not in unique_file_hashes:
                                pdf_file = BytesIO(pdf_bytes)
                                filename = os.path.basename(pdf_link)
                                process_pdf(username, pdf_file, filehash, filename )
                                st.success(f"Processed PDF: {filename}")
                                unique_file_hashes.add(filehash)
                    else:
                        st.error(f"Failed to download PDF: {pdf_link}")
            except Exception as e:
                st.error(f"Error processing {pdf_link}: {e}")
    st.success("Finished processing all PDF links.")

# =================== Streamlit UI ============================
is_dark_mode = st.get_option("theme.base") == "dark"

background_color = "#1E1E1E" if is_dark_mode else "#f9f9f9"
border_color = "#BB86FC" if is_dark_mode else "#fc0303"
text_color = "#E0E0E0" if is_dark_mode else "#000000"
user_background = "#333" if is_dark_mode else "#e3f2fd"
user_text_color = "#FFF" if is_dark_mode else "#000"

st.sidebar.write(f"👋 Welcome, {st.session_state.username}")

username = st.session_state.get("username")
if not username:
    st.error("You need to be logged in to upload files.")
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.clear()  # Clears all stored session data
    st.session_state.authenticated = False
    st.session_state.pop("username", None)
    st.rerun()
        
st.title("📄 AI Document Q&A and Web Scraper")

with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Configuration", "Web Scraper", "Database"])

    with tab1:
        st.header("Configuration")        
        selected_models = st.multiselect(
            "Select AI Models (Up to 3)", 
            AVAILABLE_MODELS,
            default=st.session_state.config["selected_models"],
        )
    
        with st.expander("Model Pricing"):
            for model, details in AVAILABLE_MODELS_DICT.items():
                st.write(f"**{model.split('/')[-1]}**: {details['price']}")

        st.session_state.config["vary_temperature"] = st.checkbox(
            "Vary Temperature", value=st.session_state.config.get("vary_temperature", True)
        )
        st.session_state.config["vary_top_p"] = st.checkbox(
            "Vary Top-P", value=st.session_state.config.get("vary_top_p", False)
        )
        st.session_state.config["temperature"] = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.config.get("temperature", 0.7), 0.05
        )
        st.session_state.config["top_p"] = st.slider(
            "Top-P", 0.0, 1.0, st.session_state.config.get("top_p", 0.9), 0.05
        )
        st.session_state.config["system_prompt"] = st.text_area(
            "System Prompt", value=st.session_state.config.get(
                "system_prompt", "You are a helpful assistant. Answer questions based on the provided context."
            )
        )

        if "config_uploader_key" not in st.session_state:
            st.session_state.config_uploader_key = 0
        
        config_file = st.file_uploader("Upload Configuration", type=['json'], key=f"config_uploader_{st.session_state.config_uploader_key}")
        if config_file:
            load_config(config_file)
            st.session_state.config_uploader_key += 1
            st.rerun()

        st.download_button("Download Config", data=save_config(st.session_state.config), file_name="config.json", mime="application/json")

            
    with tab2:
        st.header("Web Scraper")

        base_url = st.text_input("Enter Base URL", "https://www.imy.se")
        listing_endpoint = st.text_input("Enter Listing Endpoint", "tillsyner")
        pagination_format = st.text_input("Enter Pagination Format", "?query=&page=")
        num_pages = st.number_input("Enter Number of Pages", 1, 10, 3)

        if st.button("Start Scraping"):

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            with st.spinner("Scraping in progress..."):
                urls = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

            if urls:
                st.success(f"Found {len(urls)} items!")

                scrape_results = []
                with st.spinner("Scraping in progress..."):
                    scrape_results = asyncio.run(main(urls))

                pdf_links = set()
                for i, extracted_data in enumerate(scrape_results):
                    pdf_links.update(extracted_data)

                pdf_links = list(pdf_links)

                if pdf_links:
                    st.write("**Extracted PDFs:**")
                    for pdf in pdf_links:
                        st.markdown(pdf)

            else:
                st.warning("No items found.")

            if pdf_links:
                asyncio.run(store_in_DB(username, pdf_links))

    with tab3:
        st.subheader("📂 Stored Files in Database")
        st.header("📤 Upload PDFs")
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = 0

        pdf_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
        if pdf_files:
            unique_file_hashes = set(
                item.get("filehash") for item in users_collection.find({"username": username}, {"filehash": 1})
                if "filehash" in item
            )

            for pdf_file in pdf_files:
                file_hash = hashlib.md5(pdf_file.getvalue()).hexdigest()
                if file_hash in unique_file_hashes:
                    st.warning(f"⚠️ {pdf_file.name} already exists. Skipping...")
                    continue

                summary, chunks = process_pdf(username, pdf_file, file_hash, pdf_file.name)
                unique_file_hashes.add(file_hash)
                st.success(f"Processed {pdf_file.name}, extracted {len(chunks)} text chunks.")
            
            st.session_state.file_uploader_key += 1
            st.rerun()
            
        stored_files = list(users_collection.distinct("filename", {"username": username}))
        if stored_files:
            for filename in stored_files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"📄 {filename}")
                if col2.button("🗑️ Delete", key=filename):
                    delete_file(username, filename)
            if st.button("🗑️ Delete All Files"):
                delete_all_files(username)
        else:
            st.info("No files stored in the database.")

st.header("💬 Chat with Documents")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔹 **Step 1: Show Chat History Without Triggering Response Generation**
tabs = st.tabs([model.split("/")[-1] for model in selected_models])

for tab, model in zip(tabs, selected_models):
    with tab:
        model_type = AVAILABLE_MODELS_DICT[model]["type"]

        # Display stored chat history for this model
        for message in st.session_state.chat_history:
            # Show user messages for all models, but model responses only in their respective tab
            if message["model_name"] == model:
                role = "User" if message["role"] == "user" else "Model"
                message_color = user_background if message["role"] == "user" else background_color
                text_color = user_text_color if message["role"] == "user" else text_color

                if role == "Model":
                    st.markdown(f"""
                        <div style="
                            border: 2px solid {border_color}; 
                            padding: 15px; 
                            border-radius: 10px; 
                            background-color: {background_color};
                            color: {text_color};
                            margin-top: 10px;">
                            <strong style="color:#4CAF50;">Model:</strong> {model}<br>
                            <strong style="color:#FF9800;">Temperature:</strong> {message["temp"]}<br>
                            <strong style="color:#2196F3;">Top-P:</strong> {message["top_p"]}<br>
                            <hr>
                            <strong>Response:</strong> {message["text"]}
                        </div>
                    """, unsafe_allow_html=True)
                elif role == "User":
                    st.markdown(f"""
                        <div style="
                            border: 2px solid {border_color};
                            padding: 15px;
                            border-radius: 10px;
                            background-color: {message_color};
                            color: {text_color};
                            margin-bottom: 10px;">
                            <strong>{role}:</strong> {message["text"]}
                        </div>
                    """, unsafe_allow_html=True)


# 🔹 **Step 2: Generate Response Only When New Input is Entered**
if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    
    retrieved_context = []
    with st.spinner("Processing your Query..."):
        retrieved_context = retrieve_context(username, prompt)
    context = " ".join(retrieved_context) if retrieved_context else "No relevant context found."

    # Store user input only once (not per model)
    

    # Iterate through selected models and generate responses
    temp_values = [0, st.session_state.config["temperature"] / 2, st.session_state.config["temperature"]]
    top_p_values = [0, st.session_state.config["top_p"] / 2, st.session_state.config["top_p"]]

    for model in selected_models:
        model_type = AVAILABLE_MODELS_DICT[model]["type"]
        st.session_state.chat_history.append({"role": "user", "text": prompt, "model_name": model})
        for temp in temp_values if st.session_state.config["vary_temperature"] else [st.session_state.config["temperature"]]:
            for top_p in top_p_values if st.session_state.config["vary_top_p"] else [st.session_state.config["top_p"]]:
                with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
                    if model_type == "together":
                        response = generate_response(prompt, context, model, temp, top_p)
                    elif model_type == "gemini":
                        response = generate_response_gemini(prompt, context, temp, top_p)
                    elif model_type == "openai":
                        response = generate_response_openAi(prompt, context, temp, top_p)

                # Store model response in chat history with its model name
                st.session_state.chat_history.append({"role": "model", "text": response, "model_name": model, "temp": temp, "top_p": top_p})

    st.rerun()  # Force UI update to display new responses