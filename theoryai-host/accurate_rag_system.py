import os
import json
import uuid
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import requests
from collections import defaultdict

# Core libraries
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ Imports successful!")

@dataclass
class ChatMessage:
    """Structure for chat messages"""
    role: str
    content: str
    timestamp: datetime
    message_id: str = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())[:8]

@dataclass
class SearchResult:
    """Structure for search results"""
    content: str
    book_name: str
    filename: str
    score: float
    chunk_id: str
    book_id: str
    chunk_number: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class QueryAnalysis:
    """Dynamic query analysis structure"""
    query: str
    detected_books: List[str]
    query_intent: str
    complexity_level: str
    requires_quotes: bool
    expected_sources: int
    search_strategy: str
    context_requirements: Dict[str, Any]

class SimpleMemory:
    """Simplified memory for cloud deployment"""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, title: str = None) -> str:
        session_id = str(uuid.uuid4())[:12]
        if title is None:
            title = f"Chat - {datetime.now().strftime('%m/%d %H:%M')}"
        
        self.sessions[session_id] = {
            'messages': [],
            'title': title,
            'created_at': datetime.now()
        }
        return session_id
    
    def add_message(self, session_id: str, message: ChatMessage, **kwargs):
        if session_id not in self.sessions:
            self.sessions[session_id] = {'messages': [], 'title': 'Chat', 'created_at': datetime.now()}
        
        self.sessions[session_id]['messages'].append({
            'role': message.role,
            'content': message.content,
            'timestamp': message.timestamp
        })
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        if session_id not in self.sessions:
            return []
        
        messages = self.sessions[session_id]['messages'][-limit:]
        return [ChatMessage(
            role=msg['role'],
            content=msg['content'], 
            timestamp=msg['timestamp']
        ) for msg in messages]

class IntelligentPhilosophyRAG:
    """Simplified RAG system for debugging"""
    
    def __init__(self, 
                 qdrant_url: str,
                 qdrant_api_key: str,
                 collection_name: str = "psychology_books_kb",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llama_model: str = "llama3.2:3b"):
        
        print("🚀 Initializing RAG System...")
        
        self.collection_name = collection_name
        self.llama_model = llama_model
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        try:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("✅ Qdrant connected")
            
            self.embedding_model = SentenceTransformer(embedding_model)
            print("✅ Embedding model loaded")
            
            self.memory = SimpleMemory()
            print("✅ Memory initialized")
            
            # Discover books
            self.available_books = []
            self._discover_books()
            
            print(f"🎉 RAG System ready with {len(self.available_books)} books!")
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with Streamlit secrets support"""
        
        def get_config_value(key: str, default: str) -> str:
            # Try environment first
            env_val = os.getenv(key)
            if env_val:
                return env_val
            
            # Try Streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and key in st.secrets:
                    return st.secrets[key]
            except:
                pass
            
            return default
        
        return {
            'llm_base_url': get_config_value('LLM_BASE_URL', 'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf'),
            'llm_model': get_config_value('LLM_MODEL', 'meta-llama/Llama-2-7b-chat-hf'),
            'search_limit': 20,
            'max_context_length': 4000,
        }
    
    def _discover_books(self):
        """Discover available books"""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            books_found = set()
            for point in scroll_result[0]:
                book_name = point.payload.get('book_name', '').strip()
                if book_name:
                    books_found.add(book_name)
            
            self.available_books = sorted(list(books_found))
            print(f"📚 Found books: {self.available_books[:3]}..." if len(self.available_books) > 3 else f"📚 Found books: {self.available_books}")
            
        except Exception as e:
            print(f"❌ Book discovery failed: {e}")
            self.available_books = []
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face API"""
        
        # Get token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            try:
                import streamlit as st
                hf_token = st.secrets.get('HF_TOKEN')
            except:
                pass
        
        if not hf_token:
            raise Exception("No Hugging Face token found")
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        model_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.1,
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        response = requests.post(model_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                cleaned_text = generated_text.replace(formatted_prompt, '').strip()
                return cleaned_text if cleaned_text else generated_text
            else:
                return str(result)
        else:
            raise Exception(f"Hugging Face API error: {response.status_code}")
    
    def search(self, query: str) -> List[SearchResult]:
        """Simple search function"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=5,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                payload = result.payload
                results.append(SearchResult(
                    content=payload.get('content', ''),
                    book_name=payload.get('book_name', ''),
                    filename=payload.get('filename', ''),
                    score=result.score,
                    chunk_id=str(result.id),
                    book_id=payload.get('book_id', ''),
                    chunk_number=payload.get('chunk_number', 0),
                    metadata=payload
                ))
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def generate_response(self, query: str, context: str, sources: List[str]) -> str:
        """Generate response using Hugging Face"""
        
        if not context.strip():
            return "I don't have relevant information about this topic in my knowledge base."
        
        try:
            prompt = f"""Based on the following source material, answer the user's question.

Source Material:
{context}

User Question: {query}

Provide a comprehensive answer based only on the source material above."""
            
            response = self._call_huggingface(prompt)
            
            if sources:
                response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            return response
            
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return f"I found relevant information in {', '.join(sources)} but encountered technical difficulties."
    
    def chat(self, query: str, session_id: str = None) -> Tuple[str, str, List[str]]:
        """Main chat function"""
        
        try:
            # Create session if needed
            if session_id is None:
                session_id = self.memory.create_session()
            
            # Add user message
            user_message = ChatMessage(
                role="user",
                content=query,
                timestamp=datetime.now()
            )
            self.memory.add_message(session_id, user_message)
            
            # Search for relevant content
            search_results = self.search(query)
            
            if not search_results:
                response = "This information is not present in my knowledge base."
                sources = []
            else:
                # Build context
                context_parts = []
                sources = []
                seen_books = set()
                
                for i, result in enumerate(search_results):
                    book_name = result.book_name
                    if book_name not in seen_books:
                        sources.append(book_name)
                        seen_books.add(book_name)
                    
                    context_parts.append(f"[Source {i+1}: {book_name}]\n{result.content}")
                
                context = "\n\n---\n\n".join(context_parts)
                
                # Generate response
                response = self.generate_response(query, context, sources)
            
            # Add assistant response
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now()
            )
            self.memory.add_message(session_id, assistant_message)
            
            return response, session_id, sources
            
        except Exception as e:
            print(f"❌ Chat failed: {e}")
            error_response = f"I encountered an error: {str(e)}"
            return error_response, session_id or "error", []
    
    def get_available_books(self) -> List[str]:
        """Get available books"""
        return self.available_books

# Factory function
def create_intelligent_philosophy_rag(qdrant_url: str, qdrant_api_key: str, 
                                    collection_name: str = "psychology_books_kb",
                                    embedding_model: str = "all-MiniLM-L6-v2",
                                    llama_model: str = "llama3.2:3b") -> IntelligentPhilosophyRAG:
    """Create RAG system"""
    return IntelligentPhilosophyRAG(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llama_model=llama_model
    )

print("✅ accurate_rag_system.py loaded successfully!")
