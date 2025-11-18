import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
import time

# Initialize splash screen state
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

# Show splash screen animation
if not st.session_state.splash_shown:
    st.markdown("""
        <style>
        body {
            overflow: hidden;
        }
        .splash-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 999999;
        }
        
        .logo-animation {
            font-size: 120px;
            animation: bounce 1s ease-in-out infinite alternate;
        }
        
        .app-name {
            color: white;
            font-size: 48px;
            font-weight: bold;
            margin-top: 20px;
            opacity: 0;
            animation: fadeInText 1s ease-in-out 0.5s forwards;
        }
        
        .tagline {
            color: rgba(255, 255, 255, 0.9);
            font-size: 20px;
            margin-top: 10px;
            opacity: 0;
            animation: fadeInText 1s ease-in-out 1s forwards;
        }
        
        .loading-dots {
            color: white;
            font-size: 24px;
            margin-top: 30px;
            opacity: 0;
            animation: fadeInText 1s ease-in-out 1.5s forwards;
        }
        
        @keyframes bounce {
            0% {
                transform: translateY(0) scale(1);
            }
            100% {
                transform: translateY(-20px) scale(1.1);
            }
        }
        
        @keyframes fadeInText {
            to {
                opacity: 1;
            }
        }
        
        .dot {
            animation: blink 1.4s infinite;
        }
        
        .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes blink {
            0%, 20% {
                opacity: 0;
            }
            50% {
                opacity: 1;
            }
        }
        </style>
        
        <div class="splash-screen">
            <div class="logo-animation">📚</div>
            <div class="app-name">StudyMate</div>
            <div class="tagline">Your AI-Powered Academic Assistant</div>
            <div class="loading-dots">
                <span class="dot">.</span>
                <span class="dot">.</span>
                <span class="dot">.</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Wait for animation to complete
    time.sleep(3)
    st.session_state.splash_shown = True
    st.rerun()

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        animation: fadeIn 0.5s;
    }
    .user-message {
    background-color: #d9caff !important;   
    border-left: 4px solid #7b42f5 !important; 
    color: #000000 !important;             
    }
    .assistant-message {
    background-color: #ffffff;
    border-left: 4px solid #4CAF50;
    color: #000000 !important;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stats-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Functions
import pdfplumber

import pdfplumber

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""



def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into smaller chunks for better processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def load_models(hf_token: str):
    """Load the IBM Granite model and embedding model"""
    try:
        with st.spinner("🔄 Loading AI models... This may take a few moments."):
            # Load embedding model for semantic search
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load IBM Granite 3.2 2B Instruct model
            model_name = "ibm-granite/granite-3.0-2b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                model = model.to('cuda')
            
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.embedding_model = embedding_model
            
        st.success("✅ Models loaded successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("💡 Please ensure your Hugging Face token has access to the IBM Granite model.")
        return False

def create_embeddings_and_index(texts: List[str]):
    """Create embeddings and FAISS index for semantic search"""
    if st.session_state.embedding_model is None:
        st.error("Please load models first!")
        return
    
    with st.spinner("🔍 Creating embeddings for your documents..."):
        embeddings = st.session_state.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        st.session_state.embeddings = embeddings
        st.session_state.faiss_index = index
    
    st.success(f"✅ Indexed {len(texts)} text chunks!")

def retrieve_relevant_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from uploaded PDFs using semantic search"""
    if st.session_state.faiss_index is None or st.session_state.embedding_model is None:
        return ""
    
    query_embedding = st.session_state.embedding_model.encode([query])
    distances, indices = st.session_state.faiss_index.search(query_embedding.astype('float32'), k)
    
    relevant_chunks = [st.session_state.pdf_texts[i] for i in indices[0]]
    return "\n\n".join(relevant_chunks)

def generate_response(query: str, context: str) -> str:
    """Generate response using IBM Granite model"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        return "Please load the models first by entering your Hugging Face token."
    
    prompt = f"""You are StudyMate, an AI academic assistant. Answer the student's question based on the provided context from their study materials.

Context from uploaded documents:
{context}

Student's Question: {query}

Provide a clear, accurate, and well-structured answer based on the context. If the context doesn't contain enough information to answer the question, say so politely.

Answer:"""
    
    try:
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id
            )
        
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main UI
st.markdown("""
    <div class="header-container">
        <h1 style='margin:0; font-size: 2.5rem;'>📚 StudyMate</h1>
        <p style='margin:0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;'>Your AI-Powered Academic Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Hugging Face Token Input
    hf_token = st.text_input(
        "🔑 Hugging Face Access Token",
        type="password",
        help="Enter your Hugging Face access token to use IBM Granite 3.2 2B Instruct model"
    )
    
    if hf_token and st.session_state.model is None:
        if st.button("🚀 Load AI Models", use_container_width=True):
            load_models(hf_token)
    
    st.markdown("---")
    
    # PDF Upload Section
    st.markdown("### 📄 Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your textbooks, lecture notes, or research papers"
    )
    
    if uploaded_files and st.button("📥 Process PDFs", use_container_width=True):
        if st.session_state.embedding_model is None:
            st.warning("⚠️ Please load the AI models first!")
        else:
            all_chunks = []
            progress_bar = st.progress(0)
            
            for idx, pdf_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {pdf_file.name}..."):
                    text = extract_text_from_pdf(pdf_file)
                    if text:
                        chunks = chunk_text(text)
                        all_chunks.extend(chunks)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if all_chunks:
                st.session_state.pdf_texts = all_chunks
                create_embeddings_and_index(all_chunks)
    
    # Statistics
    if st.session_state.pdf_texts:
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="stats-box">
                    <h3 style='color: #667eea; margin:0;'>{len(st.session_state.pdf_texts)}</h3>
                    <p style='margin:0; color: #666;'>Text Chunks</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stats-box">
                    <h3 style='color: #764ba2; margin:0;'>{len(st.session_state.messages)}</h3>
                    <p style='margin:0; color: #666;'>Messages</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Chat with Your Study Materials")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 StudyMate:</strong><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("### 💡 Quick Tips")
    st.info("""
    **How to use:**
    1. Enter your HF token
    2. Load the AI models
    3. Upload your PDFs
    4. Ask questions!
    
    **Example questions:**
    - "Summarize chapter 3"
    - "Explain the concept of..."
    - "What are the key points about...?"
    """)
    # ---------------------- 🎤 VOICE INPUT WITH AI RESPONSE ---------------------- #
import speech_recognition as sr

st.markdown("### 🎤 Voice Input")

# Initialize state flag
if "voice_processed" not in st.session_state:
    st.session_state.voice_processed = False

audio_file = st.audio_input("Speak your question")

# Only process if new audio AND not processed before
if audio_file and not st.session_state.voice_processed:
    st.info("⏳ Converting speech to text…")

    try:
        # Convert speech → text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            voice_text = recognizer.recognize_google(audio_data)

        st.success(f"✅ You said: {voice_text}")

        # Add user message
        st.session_state.messages.append({"role": "user", "content": voice_text})

        # -------------------- AI RESPONSE -------------------- #
        if st.session_state.pdf_texts:
            with st.spinner("🤔 Thinking..."):
                context = retrieve_relevant_context(voice_text)
                response = generate_response(voice_text, context)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Mark processed so it won't loop
        st.session_state.voice_processed = True

        st.rerun()

    except Exception as e:
        st.error(f"Speech recognition error: {e}")

# Reset flag when no audio is present
if not audio_file:
    st.session_state.voice_processed = False

# Chat input
if st.session_state.model is not None and st.session_state.pdf_texts:
    user_question = st.chat_input("Ask a question about your study materials...")
    
    if user_question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Retrieve relevant context
        context = retrieve_relevant_context(user_question)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(user_question, context)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
elif st.session_state.model is None:
    st.warning("Please enter your Hugging Face token.")
elif not st.session_state.pdf_texts:
    st.warning("Please upload and process your PDF files.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎓 StudyMate - Empowering students with AI-powered learning assistance</p>
        <p style='font-size: 0.9rem;'>Powered by IBM Granite 3.2 2B Instruct</p>
    </div>
""", unsafe_allow_html=True)
