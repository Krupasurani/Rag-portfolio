import streamlit as st
from datetime import datetime
import warnings
import json
import uuid
import sys
import os

warnings.filterwarnings("ignore")

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import RAG system with better error handling
try:
    from accurate_rag_system import create_intelligent_philosophy_rag
    RAG_IMPORT_SUCCESS = True
except ImportError as e:
    RAG_IMPORT_SUCCESS = False
    st.error(f"❌ Import Error: {str(e)}")
    st.error("📁 Please ensure 'accurate_rag_system.py' is in the same directory as this app")
    st.info("🔍 Current directory contents:")
    
    # Show current directory contents for debugging
    try:
        current_files = os.listdir(current_dir)
        st.write(f"Files in {current_dir}:")
        for file in current_files:
            st.write(f"  - {file}")
    except Exception:
        st.write("Could not list directory contents")
    
    st.stop()

# Page config
st.set_page_config(
    page_title="Philosophy AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment check for Streamlit Cloud
def check_environment():
    """Check if we're running on Streamlit Cloud and show debug info"""
    if os.getenv('STREAMLIT_CLOUD') or 'streamlit.app' in os.getenv('HOSTNAME', ''):
        st.sidebar.info("🌐 Running on Streamlit Cloud")
    else:
        st.sidebar.info("💻 Running locally")

# CSS (exact same as original)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        
        --text-primary: #1a1a1a;
        --text-secondary: #4a5568;
        --text-tertiary: #718096;
        
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        
        --border-primary: #e2e8f0;
        --border-accent: #667eea;
        
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        
        --sidebar-bg: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        --sidebar-text-primary: #f8fafc;
        --sidebar-text-secondary: #cbd5e1;
        --sidebar-border: #334155;
        
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-6: 1.5rem;
        --space-8: 2rem;
        
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        --radius-full: 9999px;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f8fafc;
            --text-secondary: #e2e8f0;
            --text-tertiary: #cbd5e1;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --border-primary: #334155;
        }
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-sizing: border-box;
    }

    .stDeployButton,
    .stDecoration,
    header[data-testid="stHeader"],
    .stToolbar,
    .stActionButton {
        display: none !important;
    }

    .stApp {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    .block-container {
        background: var(--bg-primary) !important;
        padding: 0 !important;
        max-width: none !important;
    }

    .css-1d391kg,
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }

    .css-1d391kg .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--sidebar-text-primary) !important;
    }

    .css-1d391kg h3,
    [data-testid="stSidebar"] h3 {
        color: var(--sidebar-text-primary) !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        margin: var(--space-6) 0 var(--space-3) 0 !important;
        border-bottom: 1px solid var(--sidebar-border) !important;
        padding-bottom: var(--space-2) !important;
    }

    .css-1d391kg p,
    [data-testid="stSidebar"] p {
        color: var(--sidebar-text-secondary) !important;
        font-size: 0.875rem !important;
    }

    .stButton > button {
        width: 100% !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-3) var(--space-4) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        border: 1px solid var(--border-primary) !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
        margin-bottom: var(--space-2) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
        border-color: var(--border-accent) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow-md) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--secondary-gradient) !important;
        box-shadow: var(--shadow-lg) !important;
        transform: translateY(-2px) !important;
    }

    .css-1d391kg .stButton > button,
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--sidebar-text-primary) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .css-1d391kg .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(79, 172, 254, 0.5) !important;
    }

    .css-1d391kg .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: var(--sidebar-text-primary) !important;
        border-radius: var(--radius-md) !important;
        padding: var(--space-2) var(--space-3) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(79, 172, 254, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1) !important;
        outline: none !important;
    }

    .css-1d391kg .stMetric,
    [data-testid="stSidebar"] .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        padding: var(--space-3) !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: var(--sidebar-text-primary) !important;
    }

    .chat-header {
        text-align: center;
        padding: var(--space-4) var(--space-4);
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border-primary);
    }

    .chat-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .messages {
        max-width: 800px;
        margin: 0 auto;
        padding: var(--space-4) var(--space-6) 140px var(--space-6);
        background: var(--bg-primary);
    }

    .message {
        margin-bottom: var(--space-6);
    }

    .message-user {
        text-align: right;
    }

    .message-assistant {
        text-align: left;
    }

    .message-content {
        display: inline-block;
        max-width: 85%;
        padding: var(--space-4) var(--space-6);
        border-radius: var(--radius-2xl);
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: var(--shadow-md);
        word-wrap: break-word;
    }

    .user-content {
        background: var(--primary-gradient);
        color: white;
        border-bottom-right-radius: var(--radius-sm);
    }

    .assistant-content {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-primary);
        border-bottom-left-radius: var(--radius-sm);
    }

    .sources {
        margin-top: var(--space-3);
        padding: var(--space-3) var(--space-4);
        background: var(--success-gradient);
        color: white;
        border-radius: var(--radius-xl);
        font-size: 0.875rem;
        max-width: 85%;
        box-shadow: var(--shadow-md);
    }

    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 300px !important;
        right: 0 !important;
        padding: var(--space-4) !important;
        background: var(--bg-primary) !important;
        border-top: 1px solid var(--border-primary) !important;
        z-index: 1000 !important;
    }

    .stChatInput > div {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    .stChatInput input {
        border: 2px solid var(--border-primary) !important;
        border-radius: var(--radius-full) !important;
        padding: var(--space-4) var(--space-6) !important;
        font-size: 1rem !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-lg) !important;
        width: 100% !important;
    }

    .stChatInput input:focus {
        border-color: var(--border-accent) !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1), var(--shadow-lg) !important;
    }

    .stChatInput input::placeholder {
        color: var(--text-tertiary) !important;
    }

    @media (max-width: 768px) {
        .stChatInput {
            left: 0 !important;
        }
        
        .messages {
            padding: var(--space-2) var(--space-3) 140px var(--space-3);
        }
        
        .message-content {
            max-width: 95%;
        }
    }

    .stSuccess {
        background: var(--success-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-lg) !important;
    }

    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Session state and helper functions
def load_chat_sessions():
    """Load chat sessions from file"""
    try:
        if os.path.exists("chat_sessions.json"):
            with open("chat_sessions.json", "r", encoding='utf-8') as f:
                data = json.load(f)
                for session_id, session_data in data.items():
                    session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
                    for message in session_data['messages']:
                        message['timestamp'] = datetime.fromisoformat(message['timestamp'])
                return data
    except Exception as e:
        print(f"Error loading chat sessions: {e}")
    return {}

def save_chat_sessions():
    """Save chat sessions to file"""
    try:
        data_to_save = {}
        for session_id, session_data in st.session_state.chat_sessions.items():
            data_to_save[session_id] = {
                'messages': [],
                'title': session_data['title'],
                'created_at': session_data['created_at'].isoformat()
            }
            for message in session_data['messages']:
                data_to_save[session_id]['messages'].append({
                    'role': message['role'],
                    'content': message['content'],
                    'sources': message.get('sources', []),
                    'timestamp': message['timestamp'].isoformat()
                })

        with open("chat_sessions.json", "w", encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat sessions: {e}")

# Session state initialization
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'books' not in st.session_state:
    st.session_state.books = []
if 'renaming_session' not in st.session_state:
    st.session_state.renaming_session = None

# Initialize RAG system with better error handling for Streamlit Cloud
def get_or_create_rag_system():
    if st.session_state.rag_system is None:
        with st.spinner("🔄 Initializing Philosophy AI with Groq Llama 3.1 8B Instant..."):
            try:
                st.session_state.rag_system = create_intelligent_philosophy_rag(
                    qdrant_url="https://fb9ece4c-0f7a-4ec8-8ed9-dd5056f41da6.europe-west3-0.gcp.cloud.qdrant.io:6333",
                    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.AXcpbBYr2Jt5t2ysC1z0I2duz-uSK5H9cLPg5YXw_7k",
                    collection_name="psychology_books_kb",
                    llama_model="llama-3.1-8b-instant"
                )
                st.session_state.books = st.session_state.rag_system.get_available_books()
                st.success("✅ Philosophy AI initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {str(e)}")
                st.info("💡 Make sure your Groq API key is set in Streamlit Cloud secrets")
                # Create a dummy system to prevent crashes
                st.session_state.rag_system = "error"
                st.session_state.books = []
    return st.session_state.rag_system

# Helper functions
def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    st.session_state.chat_sessions[session_id] = {
        'messages': [],
        'title': 'New Chat',
        'created_at': datetime.now()
    }
    st.session_state.current_session_id = session_id
    save_chat_sessions()
    return session_id

def get_current_messages():
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions:
        return st.session_state.chat_sessions[st.session_state.current_session_id]['messages']
    return []

def add_message(role, content, sources=None):
    if not st.session_state.current_session_id:
        create_new_session()

    message = {
        'role': role,
        'content': content,
        'sources': sources or [],
        'timestamp': datetime.now()
    }

    st.session_state.chat_sessions[st.session_state.current_session_id]['messages'].append(message)

    if role == 'user' and len(st.session_state.chat_sessions[st.session_state.current_session_id]['messages']) == 1:
        title = content[:30] + "..." if len(content) > 30 else content
        st.session_state.chat_sessions[st.session_state.current_session_id]['title'] = title

    save_chat_sessions()

def rename_session(session_id, new_title):
    if session_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[session_id]['title'] = new_title
        save_chat_sessions()

# Sidebar
with st.sidebar:
    # Environment check
    check_environment()
    
    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.markdown("### 💬 Chat History")
    if st.session_state.chat_sessions:
        for session_id, session_data in reversed(list(st.session_state.chat_sessions.items())):
            is_current = session_id == st.session_state.current_session_id

            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                title = session_data.get('title', 'New Chat')
                if not title or title.strip() == '':
                    title = 'New Chat'

                if st.button(
                    title, 
                    key=f"chat_{session_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()

            with col2:
                if st.button(
                    "✏️", 
                    key=f"rename_{session_id}",
                    help="Rename conversation",
                    use_container_width=True
                ):
                    st.session_state.renaming_session = session_id
                    st.rerun()

            with col3:
                if st.button(
                    "🗑️", 
                    key=f"delete_{session_id}",
                    help="Delete conversation",
                    use_container_width=True
                ):
                    del st.session_state.chat_sessions[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    save_chat_sessions()
                    st.rerun()

            if st.session_state.renaming_session == session_id:
                with st.container():
                    new_title = st.text_input(
                        "New title:",
                        value=session_data['title'],
                        key=f"new_title_{session_id}",
                        placeholder="Enter new conversation title"
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("💾 Save", key=f"save_{session_id}", use_container_width=True):
                            if new_title.strip():
                                rename_session(session_id, new_title.strip())
                            st.session_state.renaming_session = None
                            st.rerun()

                    with col_cancel:
                        if st.button("❌ Cancel", key=f"cancel_{session_id}", use_container_width=True):
                            st.session_state.renaming_session = None
                            st.rerun()
    else:
        st.write("🌟 Start your first conversation!")

    st.markdown("### 📚 Available Books")
    if st.session_state.books:
        st.markdown("**Philosophy Collection:**")
        for i, book in enumerate(st.session_state.books, 1):
            st.markdown(f"{i}. {book}")
    else:
        if st.session_state.rag_system is None:
            st.markdown("*Books will load when you start chatting*")
        else:
            st.markdown("*Loading books...*")

    st.markdown("### 📊 System Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Books", len(st.session_state.books))
    with col2:
        st.metric("💬 Chats", len(st.session_state.chat_sessions))

# Main content
st.markdown("""
<div class="chat-header">
    <h1 class="chat-title">🧠 Philosophy AI</h1>
    <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 0.5rem;">Powered by Groq Llama 3.1 8B Instant</p>
</div>
""", unsafe_allow_html=True)

messages = get_current_messages()

if messages:
    st.markdown('<div class="messages">', unsafe_allow_html=True)
    for message in messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="message message-user">
                <div class="message-content user-content">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if message.get('sources'):
                sources_list = ", ".join(message['sources'])
                sources_html = f'<div class="sources">📚 <strong>Sources:</strong> {sources_list}</div>'

            st.markdown(f"""
            <div class="message message-assistant">
                <div class="message-content assistant-content">
                    {message['content']}
                </div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me about philosophy..."):
    rag = get_or_create_rag_system()

    add_message("user", prompt)

    if rag != "error":
        with st.spinner("🦙 Groq Llama 3.1 8B is processing your question..."):
            try:
                response, session_id, sources = rag.chat(prompt, st.session_state.current_session_id)
                add_message("assistant", response, sources)
            except Exception as e:
                add_message("assistant", f"I apologize, but I encountered an error while processing your question: {str(e)}")
    else:
        add_message("assistant", "❌ The RAG system failed to initialize. Please check the configuration and try again.")

    st.rerun()
