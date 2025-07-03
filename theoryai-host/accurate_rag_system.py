import os
import json
import uuid
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

# Core libraries
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Groq SDK with proper import handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️ Groq SDK not installed. Install with: pip install groq")
    Groq = None
    GROQ_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class GroqClient:
    """Simple Groq client wrapper to avoid proxies issue"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        
    def chat_completions_create(self, model: str, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1024):
        """Create chat completion using requests"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

class LLMPoweredAnalyzer:
    """Use LLM intelligence with custom Groq client"""
    
    def __init__(self, llm_config: Dict[str, Any], available_books: List[str]):
        self.llm_config = llm_config
        self.available_books = available_books
        self.groq_client = self._init_groq_client()
    
    def _init_groq_client(self):
        """Initialize custom Groq client"""
        try:
            api_key = self._get_groq_api_key()
            if not api_key:
                logger.error("No Groq API key found")
                return None
            
            client = GroqClient(api_key)
            
            # Test the connection
            try:
                response = client.chat_completions_create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("✅ Custom Groq client initialized successfully")
                return client
            except Exception as test_error:
                logger.error(f"Groq client test failed: {test_error}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return None
    
    def _get_groq_api_key(self) -> Optional[str]:
        """Get Groq API key from various sources"""
        # Try environment variable first
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            return api_key
        
        # Try Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                return st.secrets['GROQ_API_KEY']
        except Exception:
            pass
        
        return None
    
    def analyze_query_with_llm(self, query: str) -> QueryAnalysis:
        """Let LLM analyze the query with fallback"""
        if not self.groq_client:
            logger.warning("Groq client not available, using fallback analysis")
            return self._fallback_analysis(query)
        
        analysis_prompt = self._build_analysis_prompt(query)
        
        try:
            llm_response = self._call_groq_llm(analysis_prompt)
            return self._parse_analysis_response(llm_response, query)
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(query)
    
    def _build_analysis_prompt(self, query: str) -> str:
        """Build dynamic analysis prompt"""
        books_list = "\n".join([f"- {book}" for book in self.available_books])
        
        return f"""Analyze this user query and respond with a JSON object.

Available books: {books_list}

User Query: "{query}"

Respond with this JSON structure:
{{
    "detected_books": ["relevant book titles"],
    "query_intent": "explanation/definition/comparison/etc",
    "complexity_level": "simple/moderate/complex",
    "requires_quotes": true/false,
    "expected_sources": 3,
    "search_strategy": "broad/targeted",
    "context_requirements": {{"focus_areas": ["key concepts"]}}
}}

Only respond with the JSON, no additional text."""
    
    def _call_groq_llm(self, prompt: str) -> str:
        """Call Groq LLM using custom client"""
        if not self.groq_client:
            raise Exception("Groq client not initialized")
        
        try:
            response = self.groq_client.chat_completions_create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise Exception(f"Groq API call failed: {str(e)}")
    
    def _parse_analysis_response(self, llm_response: str, original_query: str) -> QueryAnalysis:
        """Parse LLM analysis response with fallback"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(llm_response)
            
            return QueryAnalysis(
                query=original_query,
                detected_books=analysis_data.get('detected_books', []),
                query_intent=analysis_data.get('query_intent', 'general'),
                complexity_level=analysis_data.get('complexity_level', 'moderate'),
                requires_quotes=analysis_data.get('requires_quotes', False),
                expected_sources=analysis_data.get('expected_sources', 3),
                search_strategy=analysis_data.get('search_strategy', 'broad'),
                context_requirements=analysis_data.get('context_requirements', {})
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_analysis(original_query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Simple fallback when LLM analysis fails"""
        query_lower = query.lower()
        detected_books = [book for book in self.available_books if book.lower() in query_lower]
        
        return QueryAnalysis(
            query=query,
            detected_books=detected_books,
            query_intent='general',
            complexity_level='moderate',
            requires_quotes='quote' in query_lower or 'exact' in query_lower,
            expected_sources=len(detected_books) if detected_books else 3,
            search_strategy='targeted' if detected_books else 'broad',
            context_requirements={'focus_areas': []}
        )

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
    """RAG system with custom Groq integration"""
    
    def __init__(self, 
                 qdrant_url: str,
                 qdrant_api_key: str,
                 collection_name: str = "psychology_books_kb",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llama_model: str = "llama-3.1-8b-instant"):
        
        self.collection_name = collection_name
        self.llama_model = llama_model
        
        # Load configuration
        self.config = self._load_environment_config()
        
        logger.info("🚀 Initializing Intelligent Dynamic RAG System with Custom Groq Client...")
        
        # Initialize components
        try:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            self.embedding_model = SentenceTransformer(embedding_model)
            self.memory = SimpleMemory()
            
            # Initialize custom Groq client
            self.groq_client = self._init_groq_client()
            
            # Dynamic components
            self.available_books = []
            self.book_metadata = {}
            self.analyzer = None
            
            # Initialize everything dynamically
            self._discover_knowledge_base()
            self._initialize_llm_analyzer()
            
            logger.info("🎉 Intelligent RAG System ready with Custom Groq Client!")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _init_groq_client(self):
        """Initialize custom Groq client"""
        try:
            api_key = self._get_groq_api_key()
            if not api_key:
                logger.error("No Groq API key found")
                return None
            
            client = GroqClient(api_key)
            
            # Test the connection
            try:
                response = client.chat_completions_create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("✅ Custom Groq client initialized successfully")
                return client
            except Exception as test_error:
                logger.error(f"Groq client test failed: {test_error}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return None
    
    def _get_groq_api_key(self) -> Optional[str]:
        """Get Groq API key from various sources"""
        # Try environment variable first
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            return api_key
        
        # Try Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                return st.secrets['GROQ_API_KEY']
        except Exception:
            pass
        
        return None
    
    def _load_environment_config(self) -> Dict[str, Any]:
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
            # Search settings
            'search_limit': int(get_config_value('SEARCH_LIMIT', '20')),
            'relevance_threshold': float(get_config_value('RELEVANCE_THRESHOLD', '0.1')),
            'max_context_length': int(get_config_value('MAX_CONTEXT_LENGTH', '4000')),
            'chunk_overlap_range': int(get_config_value('CHUNK_OVERLAP_RANGE', '3')),
            
            # LLM settings - Groq
            'llm_base_url': get_config_value('LLM_BASE_URL', 'https://api.groq.com'),
            'llm_model': get_config_value('LLM_MODEL', 'llama-3.1-8b-instant'),
            'llm_temperature': float(get_config_value('LLM_TEMPERATURE', '0.05')),
            'llm_timeout': int(get_config_value('LLM_TIMEOUT', '300')),
            'max_response_tokens': int(get_config_value('MAX_RESPONSE_TOKENS', '1024')),
            
            # Performance settings
            'batch_size': int(get_config_value('BATCH_SIZE', '1000')),
            'max_scan_limit': int(get_config_value('MAX_SCAN_LIMIT', '50000')),
            'enable_caching': bool(get_config_value('ENABLE_CACHING', '1') == '1'),
            
            # Quality settings
            'min_chunk_relevance': float(get_config_value('MIN_CHUNK_RELEVANCE', '0.08')),
            'max_books_per_response': int(get_config_value('MAX_BOOKS_PER_RESPONSE', '5')),
            'enable_hallucination_check': bool(get_config_value('ENABLE_HALLUCINATION_CHECK', '1') == '1'),
        }
    
    def _discover_knowledge_base(self):
        """Discover everything about the knowledge base dynamically"""
        logger.info("🔍 Discovering knowledge base structure...")
        
        try:
            books_found = set()
            book_stats = defaultdict(lambda: {'chunks': 0, 'total_chars': 0, 'sample_content': []})
            
            offset = 0
            batch_size = self.config['batch_size']
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points = scroll_result[0]
                    if not points:
                        break
                    
                    for point in points:
                        payload = point.payload
                        book_name = payload.get('book_name', '').strip()
                        content = payload.get('content', '').strip()
                        
                        if book_name and content:
                            books_found.add(book_name)
                            stats = book_stats[book_name]
                            stats['chunks'] += 1
                            stats['total_chars'] += len(content)
                            
                            # Keep sample content for analysis
                            if len(stats['sample_content']) < 3:
                                stats['sample_content'].append(content[:200])
                    
                    offset += len(points)
                    if offset >= self.config['max_scan_limit']:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error in knowledge base discovery batch: {e}")
                    break
            
            self.available_books = sorted(list(books_found))
            
            # Build metadata
            for book_name in self.available_books:
                stats = book_stats[book_name]
                self.book_metadata[book_name] = {
                    'chunk_count': stats['chunks'],
                    'avg_chunk_size': stats['total_chars'] // max(stats['chunks'], 1),
                    'sample_content': stats['sample_content']
                }
            
            logger.info(f"✅ Discovered {len(self.available_books)} books with dynamic metadata")
            
        except Exception as e:
            logger.error(f"❌ Knowledge base discovery failed: {e}")
            self.available_books = []
            self.book_metadata = {}
    
    def _initialize_llm_analyzer(self):
        """Initialize LLM-powered analyzer"""
        try:
            llm_config = {
                'model': self.config['llm_model'],
                'base_url': self.config['llm_base_url']
            }
            
            self.analyzer = LLMPoweredAnalyzer(llm_config, self.available_books)
            logger.info("✅ LLM-powered analyzer initialized with Custom Groq Client")
        except Exception as e:
            logger.warning(f"LLM analyzer initialization had issues: {e}")
            # Create a basic analyzer that will use fallback methods
            self.analyzer = LLMPoweredAnalyzer({}, self.available_books)
    
    def intelligent_search(self, query: str, analysis: QueryAnalysis) -> List[SearchResult]:
        """Intelligent search with robust error handling"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Use smart search parameters based on analysis
            search_limit = self._calculate_search_limit(analysis)
            
            # Execute search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True
            )
            
            # Process and filter results
            processed_results = self._process_search_results(search_results, analysis)
            
            # Return top results
            return processed_results[:5] if len(processed_results) > 5 else processed_results
            
        except Exception as e:
            logger.error(f"❌ Intelligent search failed: {e}")
            return []
    
    def _calculate_search_limit(self, analysis: QueryAnalysis) -> int:
        """Calculate search limit with robust fallback"""
        base_limit = 20
        
        if analysis.requires_quotes:
            return min(base_limit * 2, 40)
        elif analysis.detected_books:
            return max(base_limit // 2, 10)
        elif analysis.complexity_level == 'complex':
            return min(base_limit + 10, 35)
        else:
            return base_limit
    
    def _process_search_results(self, search_results, analysis: QueryAnalysis) -> List[SearchResult]:
        """Process search results based on analysis"""
        processed = []
        
        try:
            for result in search_results:
                payload = result.payload
                book_name = payload.get('book_name', '').strip()
                
                # Filter by detected books if specified
                if analysis.detected_books and book_name:
                    if not any(detected.lower() in book_name.lower() for detected in analysis.detected_books):
                        continue
                
                processed.append(SearchResult(
                    content=payload.get('content', ''),
                    book_name=book_name,
                    filename=payload.get('filename', ''),
                    score=result.score,
                    chunk_id=payload.get('chunk_id', str(result.id)),
                    book_id=payload.get('book_id', ''),
                    chunk_number=payload.get('chunk_number', 0),
                    metadata=payload
                ))
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
        
        return processed
    
    def _call_groq_llm_for_response(self, prompt: str) -> str:
        """Call Groq LLM for response generation"""
        if not self.groq_client:
            raise Exception("Groq client not initialized")
        
        try:
            response = self.groq_client.chat_completions_create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise Exception(f"Groq API call failed: {str(e)}")
    
    def assemble_dynamic_context(self, results: List[SearchResult], analysis: QueryAnalysis) -> Tuple[str, List[str]]:
        """Assemble context using intelligent chunking"""
        if not results:
            return "", []
        
        context_parts = []
        unique_sources = []
        seen_books = set()
        max_context = self.config['max_context_length']
        
        try:
            for i, result in enumerate(results):
                book_name = result.book_name
                if book_name not in seen_books:
                    unique_sources.append(book_name)
                    seen_books.add(book_name)
                
                assembled_content = result.content
                
                # Check context length
                current_context_length = len('\n'.join(context_parts))
                if current_context_length + len(assembled_content) > max_context:
                    remaining_space = max_context - current_context_length
                    if remaining_space > 100:
                        assembled_content = assembled_content[:remaining_space - 50] + "..."
                    else:
                        break
                
                context_parts.append(f"[Source {i+1}: {book_name}]\n{assembled_content}")
        except Exception as e:
            logger.error(f"Error assembling context: {e}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context, unique_sources
    
    def generate_intelligent_response(self, query: str, context: str, sources: List[str], 
                                    analysis: QueryAnalysis, conversation_history: str = "") -> str:
        """Generate response using custom Groq client"""
        
        if not context.strip():
            return "I don't have relevant information about this topic in my knowledge base. Please ask questions related to the available philosophical works."
        
        # If Groq client is not available, provide a basic response
        if not self.groq_client:
            logger.warning("Groq client not available, providing basic response")
            return f"Based on the available sources ({', '.join(sources)}), I found relevant information but cannot generate a detailed response due to technical limitations. Please ensure your Groq API key is properly configured."
        
        try:
            # Build comprehensive prompt
            user_prompt = f"""Based on the following source material, answer the user's question comprehensively and accurately.

Source Material:
{context}

Available Sources: {', '.join(sources)}

User Question: {query}

Instructions:
- Provide a detailed, informative answer based only on the source material above
- Include specific references to the sources when relevant
- Be accurate and don't make up information not in the sources
- Make the response engaging and easy to understand
- If the sources contain philosophical concepts, explain them clearly

Answer:"""
            
            # Call custom Groq client
            response = self._call_groq_llm_for_response(user_prompt)
            
            # Add source attribution if not present
            if sources and not any(indicator in response.lower() for indicator in ['source:', 'sources:', '📚', 'from:']):
                response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I found relevant information in {', '.join(sources)} but encountered technical difficulties generating a response. Error: {str(e)}"
    
    def format_conversation_history(self, messages: List[ChatMessage]) -> str:
        """Format conversation history"""
        if not messages:
            return "No previous conversation."
        
        recent_limit = 4
        content_limit = 150
        
        recent_messages = messages[-recent_limit:]
        history_parts = []
        
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:content_limit] + "..." if len(msg.content) > content_limit else msg.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def chat(self, query: str, session_id: str = None) -> Tuple[str, str, List[str]]:
        """Main chat function with custom Groq integration"""
        
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
            
            # Get conversation history
            history = self.memory.get_conversation_history(session_id, limit=6)
            conversation_context = self.format_conversation_history(history[:-1])
            
            # LLM-powered query analysis
            if self.analyzer:
                query_analysis = self.analyzer.analyze_query_with_llm(query)
            else:
                # Fallback analysis if analyzer failed to initialize
                query_analysis = QueryAnalysis(
                    query=query,
                    detected_books=[],
                    query_intent='general',
                    complexity_level='moderate',
                    requires_quotes=False,
                    expected_sources=3,
                    search_strategy='broad',
                    context_requirements={}
                )
            
            # Intelligent search based on analysis
            search_results = self.intelligent_search(query, query_analysis)
            
            # Check if we have relevant results
            has_relevant_info = len(search_results) > 0
            
            if not has_relevant_info:
                response = "This information is not present in my knowledge base. Please ask questions related to the philosophical works I have access to."
                sources = []
            else:
                # Assemble context dynamically
                context, sources = self.assemble_dynamic_context(search_results, query_analysis)
                
                # Generate intelligent response
                response = self.generate_intelligent_response(
                    query, context, sources, query_analysis, conversation_context
                )
            
            # Add assistant response with metadata
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now()
            )
            self.memory.add_message(session_id, assistant_message)
            
            # Log interaction with analysis details
            logger.info(f"💬 Session {session_id}: Intent={query_analysis.query_intent}, "
                       f"Books={len(query_analysis.detected_books)}, Sources={len(sources)}")
            
            return response, session_id, sources
            
        except Exception as e:
            logger.error(f"Chat function failed: {e}")
            error_response = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            return error_response, session_id or "error", []
    
    def get_available_books(self) -> List[str]:
        """Get dynamically discovered books"""
        return self.available_books
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get dynamic collection statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Calculate dynamic stats
            total_chunks = sum(meta.get('chunk_count', 0) for meta in self.book_metadata.values())
            avg_chunks_per_book = total_chunks / max(len(self.available_books), 1)
            
            return {
                'total_points': collection_info.points_count or 0,
                'collection_status': collection_info.status,
                'available_books': len(self.available_books),
                'total_chunks': total_chunks,
                'avg_chunks_per_book': round(avg_chunks_per_book, 1),
                'llm_model': self.config['llm_model'],
                'system_ready': self.groq_client is not None,
                'groq_available': True
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                'total_points': 'Error',
                'collection_status': 'Unknown',
                'available_books': len(self.available_books),
                'system_ready': False,
                'groq_available': False
            }

# Factory function for easy integration
def create_intelligent_philosophy_rag(qdrant_url: str, qdrant_api_key: str, 
                                    collection_name: str = "psychology_books_kb",
                                    embedding_model: str = "all-MiniLM-L6-v2",
                                    llama_model: str = "llama-3.1-8b-instant") -> IntelligentPhilosophyRAG:
    """Create an intelligent LLM-powered philosophy RAG system with custom Groq client"""
    return IntelligentPhilosophyRAG(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llama_model=llama_model
    )
