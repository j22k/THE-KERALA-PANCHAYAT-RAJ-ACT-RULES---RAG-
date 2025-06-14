"""
Kerala Panchayat RAG Streamlit Application
Elegant Kerala-themed UI with CUDA fix and user-friendly interface
"""

import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import time
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kerala Panchayat Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Elegant Kerala-themed CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Elegant Kerala Color Palette */
    :root {
        --cream-bg: #FFFDD0;
        --dark-green: #004225;
        --muted-gold: #D4AF37;
        --cream-secondary: #F5F5DC;
        --shadow-subtle: 0 2px 8px rgba(0, 66, 37, 0.08);
        --shadow-medium: 0 4px 12px rgba(0, 66, 37, 0.12);
    }
    
    /* Base styling */
    .stApp {
        background-color: var(--cream-bg);
        font-family: 'Inter', sans-serif;
        color: var(--dark-green);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', serif;
        color: var(--dark-green) !important;
        font-weight: 400;
    }
    
    p, span, div, label {
        color: var(--dark-green) !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header - Elegant and Minimal */
    .main-header {
        background: linear-gradient(135deg, var(--cream-secondary) 0%, var(--cream-bg) 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-subtle);
        border: 1px solid rgba(212, 175, 55, 0.2);
        position: relative;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--muted-gold);
        border-radius: 12px 12px 0 0;
    }
    
    .main-header h1 {
        color: var(--dark-green) !important;
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        font-family: 'Merriweather', serif;
    }
    
    .main-header p {
        color: var(--dark-green) !important;
        opacity: 0.8;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 300;
    }
    
    /* Chat Interface */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Query Input */
    .query-section {
        background: var(--cream-secondary);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: var(--shadow-subtle);
        border-left: 4px solid var(--muted-gold);
    }
    
    .query-section h3 {
        margin-top: 0;
        color: var(--dark-green) !important;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: var(--cream-bg) !important;
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 8px !important;
        color: var(--dark-green) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--muted-gold) !important;
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(0, 66, 37, 0.6) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--muted-gold) 0%, #C4941C 100%) !important;
        color: var(--cream-bg) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        box-shadow: var(--shadow-subtle) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #C4941C 0%, var(--muted-gold) 100%) !important;
        box-shadow: var(--shadow-medium) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Answer display */
    .answer-container {
        background: var(--cream-bg);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: var(--shadow-subtle);
        position: relative;
    }
    
    .answer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--muted-gold);
        border-radius: 12px 12px 0 0;
    }
    
    .answer-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .answer-header h3 {
        margin: 0;
        color: var(--dark-green) !important;
        font-size: 1.3rem;
    }
    
    .response-time {
        font-size: 0.9rem;
        color: rgba(0, 66, 37, 0.7) !important;
        font-style: italic;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    /* Source boxes */
    .source-container {
        background: var(--cream-secondary);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 3px solid var(--muted-gold);
        box-shadow: var(--shadow-subtle);
    }
    
    .source-container strong {
        color: var(--dark-green) !important;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-bar {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(245, 245, 220, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: var(--dark-green) !important;
        box-shadow: var(--shadow-medium);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    .device-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .device-indicator::before {
        content: '🖥️';
        font-size: 1rem;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: var(--cream-secondary) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        color: var(--dark-green) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .streamlit-expanderContent {
        background: var(--cream-bg) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-top: none !important;
    }
    
    /* Info messages */
    .stSuccess {
        background-color: rgba(212, 175, 55, 0.1) !important;
        color: var(--dark-green) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
    }
    
    .stInfo {
        background-color: rgba(212, 175, 55, 0.1) !important;
        color: var(--dark-green) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
    }
    
    .stWarning {
        background-color: rgba(212, 175, 55, 0.15) !important;
        color: var(--dark-green) !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
    }
    
    .stError {
        background-color: rgba(212, 175, 55, 0.1) !important;
        color: var(--dark-green) !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Spinner */
    .stSpinner {
        color: var(--muted-gold) !important;
    }
    
    /* Footer info */
    .footer-info {
        background: var(--cream-secondary);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 3rem 0 1rem 0;
        border-left: 3px solid var(--muted-gold);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure proper text colors throughout */
    .stMarkdown, .stMarkdown *, .stText, .stText * {
        color: var(--dark-green) !important;
    }
    
    /* Button text exception */
    .stButton > button, .stButton > button * {
        color: var(--cream-bg) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .query-section, .answer-container {
            padding: 1.5rem;
        }
        
        .status-bar {
            bottom: 10px;
            right: 10px;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def detect_device():
    """Properly detect CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            # Additional check to ensure CUDA is actually working
            try:
                torch.cuda.current_device()
                return 'cuda'
            except:
                return 'cpu'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

class KeralaPanchayatRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Kerala Panchayat RAG System
        """
        # Get API key from environment
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Detect device properly
        self.device = detect_device()
        
        # Load embedding model with caching
        @st.cache_resource
        def load_embedding_model(model_name, device):
            try:
                return SentenceTransformer(model_name, device=device)
            except Exception as e:
                st.warning(f"Failed to load model on {device}, falling back to CPU")
                return SentenceTransformer(model_name, device='cpu')
        
        self.embedding_model = load_embedding_model(model_name, self.device)
        self.chunks = []
        self.index = None
    
    def load_system(self, index_path: str = "kerala_panchayat_index.bin", 
                   chunks_path: str = "kerala_chunks.pkl"):
        """Load the preprocessed FAISS index and chunks"""
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return True, f"System ready with {len(self.chunks)} sections loaded"
        except FileNotFoundError:
            return False, "Data files not found. Please run ingest_pdf.py first."
        except Exception as e:
            return False, f"Error loading system: {str(e)}"
    
    def search_relevant_sections(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant sections"""
        if self.index is None:
            raise ValueError("System not loaded")
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def answer_query(self, query: str) -> dict:
        """Generate simple, user-friendly answer"""
        start_time = time.time()
        
        try:
            relevant_sections = self.search_relevant_sections(query, k=3)
            
            if not relevant_sections:
                return {
                    'answer': "I couldn't find information about this topic in the Kerala Panchayat documents. Could you try asking in a different way?",
                    'sources': [],
                    'confidence': 0.0,
                    'response_time': time.time() - start_time
                }
            
            context = "\n\n".join([section for section, _ in relevant_sections])
            
            system_prompt = """You are a helpful assistant that explains Kerala Panchayat rules and procedures in simple, easy-to-understand language.

Your guidelines:
- Use simple, clear language that anyone can understand
- Avoid legal jargon and complex terms
- Break down complex procedures into simple steps
- Give practical examples when helpful
- Use bullet points for lists and procedures
- Be encouraging and helpful in tone
- If something is technical, explain it in simple terms first

Remember: Your users may not have legal or administrative background, so make everything easy to understand."""

            user_prompt = f"""Based on this information from Kerala Panchayat documents, please answer the user's question in simple, clear language:

REFERENCE INFORMATION:
{context}

USER'S QUESTION: {query}

Please provide a helpful answer that:
1. Explains things in simple terms
2. Uses easy-to-understand language
3. Gives step-by-step guidance if needed
4. Is encouraging and supportive

ANSWER:"""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=800,
            )
            
            answer = chat_completion.choices[0].message.content
            response_time = time.time() - start_time
            
            return {
                'answer': answer,
                'sources': relevant_sections,
                'confidence': sum(score for _, score in relevant_sections) / len(relevant_sections) if relevant_sections else 0,
                'num_sources': len(relevant_sections),
                'response_time': response_time
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error while searching for your answer: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'response_time': time.time() - start_time
            }

def render_header():
    """Elegant Kerala-themed header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Kerala Panchayat Assistant</h1>
        <p>Your guide to Panchayat rules and procedures</p>
    </div>
    """, unsafe_allow_html=True)

def render_device_status(device):
    """Render device status indicator"""
    device_display = device.upper()
    st.markdown(f"""
    <div class="status-bar">
        <div class="device-indicator">
            Device: {device_display}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Check for API key first
    if not os.getenv('GROQ_API_KEY'):
        st.error("⚠️ **Setup Required**")
        st.markdown("""
        **Please set up your API key:**
        
        1. Create a file called `.env` in your project folder
        2. Add this line to the file: `GROQ_API_KEY=your_api_key_here`
        3. Get your free API key from: https://console.groq.com/
        4. Restart the application
        
        **Need help?** The API key is free and takes 2 minutes to get!
        """)
        return
    
    # Render header
    render_header()
    
    # Check if system files exist
    if not (os.path.exists("kerala_panchayat_index.bin") and os.path.exists("kerala_chunks.pkl")):
        st.error("📁 **Data Files Missing**")
        st.markdown("""
        **Please prepare the data files first:**
        
        1. Run this command in your terminal: `python ingest_pdf.py`
        2. Wait for it to process the PDF files
        3. Refresh this page
        
        **What this does:** It prepares the Kerala Panchayat documents so the assistant can search through them.
        """)
        return
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        with st.spinner("🚀 Starting up the assistant..."):
            try:
                st.session_state.rag_system = KeralaPanchayatRAG()
                success, message = st.session_state.rag_system.load_system()
                
                if success:
                    st.success(f"✅ {message}")
                    device = st.session_state.rag_system.device
                    st.info(f"System initialized and ready")
                    # Store device for status display
                    st.session_state.device = device
                else:
                    st.error(f"❌ {message}")
                    return
                    
            except Exception as e:
                st.error(f"❌ Failed to start: {e}")
                return
    
    # Render device status
    if 'device' in st.session_state:
        render_device_status(st.session_state.device)
    
    # Main query interface - Clean start without sample questions
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    st.markdown("### Ask Your Question")
    
    query = st.text_area(
        "",
        height=120,
        placeholder="What would you like to know about Kerala Panchayats? For example: How do I apply for a certificate from my Panchayat?",
        help="Ask anything about Panchayat rules, procedures, or services",
        label_visibility="collapsed"
    )
    
    search_clicked = st.button("🔍 Get Answer", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if search_clicked and query.strip():
        with st.spinner("🔍 Searching for your answer..."):
            result = st.session_state.rag_system.answer_query(query)
            
            # Show answer
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown("""
            <div class="answer-header">
                <h3>📋 Your Answer</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(result['answer'])
            
            # Show response time
            st.markdown(f"""
            <div class="response-time">
                Responded in: {result['response_time']:.2f}s
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show sources if available
            if result['sources']:
                with st.expander("📚 Reference Sources", expanded=False):
                    st.markdown("*This answer is based on these sections from Kerala Panchayat documents:*")
                    for i, (source, score) in enumerate(result['sources'][:2]):
                        st.markdown(f"""
                        <div class="source-container">
                            <strong>Source {i+1}:</strong><br>
                            {source[:400]}{'...' if len(source) > 400 else ''}
                        </div>
                        """, unsafe_allow_html=True)
    
    elif search_clicked:
        st.warning("Please enter a question to get an answer.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer-info">
        <strong>About this Assistant:</strong><br>
        This assistant provides information based on Kerala Panchayat documents. For official matters, please contact your local Panchayat office directly. The responses are generated using AI and should be used as guidance only.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()