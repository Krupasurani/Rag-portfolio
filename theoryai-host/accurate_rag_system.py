import os
import json
import uuid
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import sqlite3
from dataclasses import dataclass, asdict
import requests
from collections import defaultdict
from functools import lru_cache

# Core libraries
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

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

class LLMPoweredAnalyzer:
    """Use LLM intelligence for all analysis instead of hardcoded patterns"""
    
    def __init__(self, llm_config: Dict[str, Any], available_books: List[str]):
        self.llm_config = llm_config
        self.available_books = available_books
    
    def analyze_query_with_llm(self, query: str) -> QueryAnalysis:
        """Let LLM analyze the query - no fallbacks"""
        analysis_prompt = self._build_analysis_prompt(query)
        
        llm_response = self._call_llm(analysis_prompt)
        return self._parse_analysis_response(llm_response, query)
    
    def _build_analysis_prompt(self, query: str) -> str:
        """Build dynamic analysis prompt"""
        books_list = "\n".join([f"- {book}" for book in self.available_books])
        
        return f"""
You are an intelligent query analyzer for a philosophy knowledge base. Analyze this user query and respond with a JSON object.

Available books in knowledge base:
{books_list}

User Query: "{query}"

Analyze and respond with this exact JSON structure:
{{
    "detected_books": ["list of book titles mentioned or relevant to the query"],
    "query_intent": "what the user wants (explanation/definition/comparison/quotation/analysis/etc)",
    "complexity_level": "simple/moderate/complex based on query depth",
    "requires_quotes": true/false,
    "expected_sources": number_of_sources_needed,
    "search_strategy": "targeted/broad/multi_book/exact_match",
    "context_requirements": {{
        "chunk_size": "small/medium/large",
        "chunk_overlap": "low/medium/high", 
        "focus_areas": ["key concepts to focus on"]
    }}
}}

Only respond with the JSON, no additional text.
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with robust error handling"""
        try:
            chat_data = {
                "model": self.llm_config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.llm_config['base_url']}/api/chat",
                json=chat_data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '')
                if content:
                    return content
                else:
                    raise Exception("Empty response from LLM")
            else:
                raise Exception(f"LLM returned status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("LLM request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to LLM server")
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _parse_analysis_response(self, llm_response: str, original_query: str) -> QueryAnalysis:
        """Parse LLM analysis response - no fallbacks"""
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
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Simple fallback when LLM analysis fails"""
        query_lower = query.lower()
        
        # Simple book detection
        detected_books = [book for book in self.available_books if book.lower() in query_lower]
        
        return QueryAnalysis(
            query=query,
            detected_books=detected_books,
            query_intent='general',
            complexity_level='moderate',
            requires_quotes='quote' in query_lower or 'exact' in query_lower,
            expected_sources=len(detected_books) if detected_books else 3,
            search_strategy='targeted' if detected_books else 'broad',
            context_requirements={'chunk_size': 'medium', 'chunk_overlap': 'medium', 'focus_areas': []}
        )

class ConversationMemory:
    """Enhanced conversation memory"""
    
    def __init__(self, db_path: str = "philosophy_chat_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    has_sources BOOLEAN DEFAULT FALSE,
                    query_analysis TEXT,
                    sources_used TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
        
        logger.info("✅ Conversation memory initialized")
    
    def create_session(self, title: str = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())[:12]
        if title is None:
            title = f"Chat - {datetime.now().strftime('%m/%d %H:%M')}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('INSERT INTO sessions (session_id, title) VALUES (?, ?)', (session_id, title))
        
        return session_id
    
    def add_message(self, session_id: str, message: ChatMessage, 
                   has_sources: bool = False, query_analysis: Dict = None, 
                   sources_used: List[str] = None):
        """Add message with metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversations 
                (session_id, message_id, role, content, timestamp, has_sources, 
                 query_analysis, sources_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, message.message_id, message.role, 
                  message.content, message.timestamp.isoformat(), has_sources,
                  json.dumps(query_analysis) if query_analysis else None,
                  json.dumps(sources_used) if sources_used else None))
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """Get conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT message_id, role, content, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            messages = []
            for row in reversed(cursor.fetchall()):
                messages.append(ChatMessage(
                    message_id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                ))
            
            return messages

class IntelligentPhilosophyRAG:
    """Fully dynamic RAG system powered by LLM intelligence"""
    
    def __init__(self, 
                 qdrant_url: str,
                 qdrant_api_key: str,
                 collection_name: str = "psychology_books_kb",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llama_model: str = "llama3.2:3b"):
        
        self.collection_name = collection_name
        self.llama_model = llama_model
        
        # Load all configuration from environment
        self.config = self._load_environment_config()
        
        logger.info("🚀 Initializing Intelligent Dynamic RAG System...")
        
        # Initialize core components
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.memory = ConversationMemory()
        
        # Dynamic components - all discovered at runtime
        self.available_books = []
        self.book_metadata = {}
        self.analyzer = None
        
        # Initialize everything dynamically
        self._discover_knowledge_base()
        self._initialize_llm_analyzer()
        
        logger.info("🎉 Intelligent RAG System ready!")
    

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load all configuration from environment variables"""

        # Helper function to get config from streamlit secrets or env
        def get_config_value(key: str, default: str) -> str:
            # First try environment variables
            env_val = os.getenv(key)
            if env_val:
                return env_val

            # Then try streamlit secrets
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

            # LLM settings - Updated for Hugging Face
            'llm_base_url': get_config_value('LLM_BASE_URL', 'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf'),
            'llm_model': get_config_value('LLM_MODEL', 'meta-llama/Llama-2-7b-chat-hf'),
            'llm_temperature': float(get_config_value('LLM_TEMPERATURE', '0.05')),
            'llm_timeout': int(get_config_value('LLM_TIMEOUT', '300')),
            'max_response_tokens': int(get_config_value('MAX_RESPONSE_TOKENS', '1500')),

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
            
            self.available_books = sorted(list(books_found))
            
            # Build metadata using LLM analysis
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
        llm_config = {
            'model': self.config['llm_model'],
            'base_url': self.config['llm_base_url']
        }
        
        self.analyzer = LLMPoweredAnalyzer(llm_config, self.available_books)
        logger.info("✅ LLM-powered analyzer initialized")
    
    def get_book_context_with_llm(self, book_name: str) -> str:
        """Get dynamic book context using LLM"""
        if book_name not in self.book_metadata:
            return ""
        
        sample_content = self.book_metadata[book_name].get('sample_content', [])
        if not sample_content:
            return ""
        
        context_prompt = f"""
        Analyze this book sample and provide a brief context description:
        
        Book: {book_name}
        Sample content: {' '.join(sample_content)}
        
        Respond with just 1-2 sentences describing the book's main subject and style.
        """
        
        try:
            response = self._call_llm_simple(context_prompt)
            return response.strip()
        except:
            return f"Book: {book_name}"
    
    def intelligent_search(self, query: str, analysis: QueryAnalysis) -> List[SearchResult]:
        """Intelligent search with robust error handling and no LLM dependencies"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Use smart search parameters based on analysis (no LLM calls)
            search_limit = self._calculate_search_limit_simple(analysis)
            
            # Execute search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True
            )
            
            # Process and filter results
            processed_results = self._process_search_results(search_results, analysis)
            
            # Skip LLM filtering if LLM is having issues - just return top results
            if len(processed_results) > 5:
                return processed_results[:5]
            else:
                return processed_results
            
        except Exception as e:
            logger.error(f"❌ Intelligent search failed: {e}")
            return []
    
    def _calculate_search_limit_simple(self, analysis: QueryAnalysis) -> int:
        """Calculate search limit using LLM intelligence - no fallbacks"""
        
        optimization_request = f"""
Query: "{analysis.query}"
Intent: {analysis.query_intent}

How many search results needed (5-40)? Respond with just a number.
"""
        
        llm_response = self._call_llm_simple(optimization_request)
        
        # Extract number from response
        number_match = re.search(r'\b([5-9]|[1-3][0-9]|40)\b', llm_response)
        if number_match:
            return int(number_match.group(1))
        else:
            # Try again with simpler prompt
            simple_request = f"For query '{analysis.query}', how many search results (5-40)? Number only:"
            retry_response = self._call_llm_simple(simple_request)
            retry_match = re.search(r'\b([5-9]|[1-3][0-9]|40)\b', retry_response)
            if retry_match:
                return int(retry_match.group(1))
            else:
                return 20  # Only if LLM completely fails to give a number
    
    def _calculate_search_limit(self, analysis: QueryAnalysis) -> int:
        """Calculate search limit with robust fallback"""
        
        try:
            # Simplified LLM request
            optimization_request = f"""
Query: "{analysis.query}"
Intent: {analysis.query_intent}

How many search results needed (5-40)? Respond with just a number.
"""
            
            llm_response = self._call_llm_simple(optimization_request)
            
            # Extract number from response
            number_match = re.search(r'\b([5-9]|[1-3][0-9]|40)\b', llm_response)
            if number_match:
                return int(number_match.group(1))
        
        except Exception as e:
            logger.warning(f"Dynamic search limit calculation failed: {e}")
        
        # Smart fallback based on analysis
        base_limit = 20
        
        if analysis.requires_quotes:
            return min(base_limit * 2, 40)  # Need more results for quotes
        elif analysis.detected_books:
            return max(base_limit // 2, 10)  # Fewer results if book is targeted
        elif analysis.complexity_level == 'complex':
            return min(base_limit + 10, 35)  # More results for complex queries
        else:
            return base_limit
    
    def _process_search_results(self, search_results, analysis: QueryAnalysis) -> List[SearchResult]:
        """Process search results based on analysis"""
        processed = []
        
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
        
        return processed
    
    def _llm_filter_relevance(self, results: List[SearchResult], query: str, analysis: QueryAnalysis) -> List[SearchResult]:
        """Use LLM to filter relevance and prevent hallucination"""
        if not results:
            return results
        
        # Take top results for LLM evaluation
        top_results = results[:min(10, len(results))]
        
        relevance_prompt = f"""
        Query: "{query}"
        Query Intent: {analysis.query_intent}
        
        Evaluate which of these search results are truly relevant to answering the user's query:
        
        {self._format_results_for_llm(top_results)}
        
        Respond with a JSON array of result numbers (1-{len(top_results)}) that are relevant, ordered by relevance.
        Example: [1, 3, 5]
        
        Only respond with the JSON array, no additional text.
        """
        
        try:
            llm_response = self._call_llm_simple(relevance_prompt)
            relevant_indices = json.loads(llm_response.strip())
            
            filtered_results = []
            for idx in relevant_indices:
                if 1 <= idx <= len(top_results):
                    filtered_results.append(top_results[idx - 1])
            
            return filtered_results
            
        except Exception as e:
            logger.warning(f"LLM relevance filtering failed: {e}")
            return results[:5]  # Fallback to top 5
    
    def _format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results for LLM evaluation"""
        formatted = []
        for i, result in enumerate(results, 1):
            content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
            formatted.append(f"{i}. [{result.book_name}] {content_preview}")
        return "\n\n".join(formatted)
    
    def _call_llm_simple(self, prompt: str) -> str:
        """Simple LLM call with Hugging Face support"""
        try:
            # Check if using Hugging Face
            if "api-inference.huggingface.co" in self.config['llm_base_url']:
                return self._call_huggingface(prompt)
            else:
                # Your original Ollama code
                chat_data = {
                    "model": self.config['llm_model'],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300,
                        "top_p": 0.9
                    }
                }

                response = requests.post(
                    f"{self.config['llm_base_url']}/api/chat",
                    json=chat_data,
                    timeout=300  
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get('message', {}).get('content', '')
                    if content:
                        return content
                    else:
                        raise Exception("Empty response from LLM")
                else:
                    raise Exception(f"LLM returned status {response.status_code}")

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            raise Exception(f"LLM call failed: {str(e)}")

    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API"""
        import os

        # Get token from environment or secrets
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            # Try to get from streamlit secrets if available
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

        # Use Llama 2 for better responses
        model_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

        # Format prompt for Llama 2 chat format
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

        response = requests.post(model_url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Clean up the response
                cleaned_text = generated_text.replace(formatted_prompt, '').strip()
                return cleaned_text if cleaned_text else generated_text
            else:
                return str(result)
        else:
            error_msg = f"Hugging Face API error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
    
    def assemble_dynamic_context(self, results: List[SearchResult], analysis: QueryAnalysis) -> Tuple[str, List[str]]:
        """Assemble context using LLM intelligence for chunk overlap decision"""
        if not results:
            return "", []
        
        context_parts = []
        unique_sources = []
        seen_books = set()
        
        # Use LLM to determine chunk overlap
        overlap_request = f"""
Query: "{analysis.query}"
Intent: {analysis.query_intent}

How many surrounding chunks for context (0-3)? Just respond with a number.
"""
        
        try:
            llm_response = self._call_llm_simple(overlap_request)
            number_match = re.search(r'\b([0-3])\b', llm_response)
            chunk_overlap = int(number_match.group(1)) if number_match else 2
        except:
            chunk_overlap = 2  # Emergency default only
        
        max_context = self.config['max_context_length']
        
        for i, result in enumerate(results):
            book_name = result.book_name
            if book_name not in seen_books:
                unique_sources.append(book_name)
                seen_books.add(book_name)
            
            # Get content (with error handling for chunk assembly)
            if chunk_overlap > 0 and analysis.complexity_level in ['moderate', 'complex']:
                assembled_content = self._get_related_chunks(result, chunk_overlap)
            else:
                assembled_content = result.content
            
            # Trim content to fit context limit
            current_context_length = len('\n'.join(context_parts))
            if current_context_length + len(assembled_content) > max_context:
                remaining_space = max_context - current_context_length
                if remaining_space > 100:  # Only add if meaningful space left
                    assembled_content = assembled_content[:remaining_space - 50] + "..."
                else:
                    break
            
            context_parts.append(f"[Source {i+1}: {book_name}]\n{assembled_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context, unique_sources
    
    def _get_chunk_overlap_from_analysis(self, analysis: QueryAnalysis) -> int:
        """Get chunk overlap with simple fallback"""
        
        try:
            # Quick LLM request
            overlap_request = f"""
Query: "{analysis.query}"
How many surrounding chunks for context (0-3)? Just respond with a number.
"""
            
            llm_response = self._call_llm_simple(overlap_request)
            number_match = re.search(r'\b([0-3])\b', llm_response)
            if number_match:
                return int(number_match.group(1))
        
        except Exception as e:
            logger.warning(f"Dynamic chunk overlap calculation failed: {e}")
        
        # Simple fallback logic
        if analysis.requires_quotes:
            return 1  # Minimal context for quotes
        elif analysis.complexity_level == 'complex':
            return 3  # More context for complex queries
        else:
            return 2  # Default moderate context
    
    def _get_related_chunks(self, result: SearchResult, overlap_range: int) -> str:
        """Get related chunks without using filters that require indexes"""
        if overlap_range <= 0:
            return result.content
            
        try:
            # Use simple scroll without filters to avoid index requirements
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,  # Get larger sample
                with_payload=True,
                timeout=300
                )
            
            points = scroll_result[0]
            
            # Filter in Python instead of using Qdrant filters
            nearby_chunks = []
            target_chunk = result.chunk_number
            target_book = result.book_name
            
            for point in points:
                payload = point.payload
                chunk_num = payload.get('chunk_number', 0)
                book_name = payload.get('book_name', '')
                
                # Check if this is from the same book and within range
                if (book_name == target_book and 
                    abs(chunk_num - target_chunk) <= overlap_range):
                    nearby_chunks.append({
                        'chunk_number': chunk_num,
                        'content': payload.get('content', '')
                    })
            
            if not nearby_chunks:
                return result.content
            
            # Sort by chunk number and combine
            nearby_chunks.sort(key=lambda x: x['chunk_number'])
            combined_content = '\n\n'.join([chunk['content'] for chunk in nearby_chunks])
            
            return combined_content if combined_content else result.content
            
        except Exception as e:
            logger.warning(f"Failed to get related chunks: {e}")
            return result.content
    
    def generate_intelligent_response(self, query: str, context: str, sources: List[str], 
                                    analysis: QueryAnalysis, conversation_history: str = "") -> str:
        """Generate response using pure LLM intelligence"""
        
        if not context.strip():
            return "I don't have relevant information about this topic in my knowledge base. Please ask questions related to the available philosophical works."
        
        # Build prompts using LLM intelligence
        system_prompt = self._build_intelligent_system_prompt(analysis, sources)
        user_prompt = self._build_intelligent_user_prompt(query, context, sources, conversation_history, analysis)
        
        # Generate response
        chat_data = {
            "model": self.config['llm_model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.config['llm_temperature'],
                "num_predict": self.config['max_response_tokens'],
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(
            f"{self.config['llm_base_url']}/api/chat",
            json=chat_data,
            timeout=300  # Longer timeout for main response
        )
        
        if response.status_code == 200:
            generated_response = response.json().get('message', {}).get('content', '')
            if generated_response:
                return self._post_process_response(generated_response, sources, analysis)
        
        # Only if complete LLM failure
        return f"I found relevant information in {', '.join(sources)} but encountered technical difficulties generating a response."
    
    def _build_intelligent_system_prompt(self, analysis: QueryAnalysis, sources: List[str]) -> str:
        """Build system prompt using LLM intelligence - no fallbacks"""
        
        prompt_generation_request = f"""
Generate a system prompt for an AI assistant answering this query:

Query: "{analysis.query}"
Intent: {analysis.query_intent}
Sources: {len(sources)} available

Create a 2-sentence system prompt focusing on how to use source material.
Respond with just the prompt text.
"""
        
        dynamic_prompt = self._call_llm_simple(prompt_generation_request)
        
        # Validate response
        if len(dynamic_prompt.strip()) > 20:  # Reasonable response
            base_safety = "You are an expert assistant with access to philosophical texts. CRITICAL: Only use information from the provided source material."
            return f"{base_safety}\n\n{dynamic_prompt.strip()}"
        
        # If response too short, use LLM again with different prompt
        retry_prompt = f"""Create a system prompt for answering: "{analysis.query}". Focus on using only provided sources. 2 sentences."""
        retry_response = self._call_llm_simple(retry_prompt)
        
        base_safety = "You are an expert assistant with access to philosophical texts. CRITICAL: Only use information from the provided source material."
        return f"{base_safety}\n\n{retry_response.strip()}"
    
    def _build_intelligent_user_prompt(self, query: str, context: str, sources: List[str], 
                                     conversation_history: str, analysis: QueryAnalysis) -> str:
        """Build user prompt using LLM intelligence - no fallbacks"""
        
        structure_request = f"""
For this query: "{query}"
Intent: {analysis.query_intent}

Respond with JSON:
{{"context_style": "detailed/summary", "include_history": true/false}}
"""
        
        structure_response = self._call_llm_simple(structure_request)
        
        # Parse LLM response
        json_match = re.search(r'\{.*\}', structure_response, re.DOTALL)
        if json_match:
            structure = json.loads(json_match.group())
        else:
            structure = json.loads(structure_response)
        
        # Use LLM recommendations
        context_to_use = context
        if structure.get('context_style') == 'summary' and len(context) > 2000:
            context_to_use = context[:2000] + "\n\n[Content truncated]"
        
        prompt_sections = [f"Source Material:\n{context_to_use}"]
        
        if sources:
            prompt_sections.append(f"Available Sources: {', '.join(sources)}")
        
        if (structure.get('include_history', False) and conversation_history and 
            "No previous conversation" not in conversation_history):
            prompt_sections.append(f"Previous Context: {conversation_history}")
        
        prompt_sections.extend([
            f"User Query: {query}",
            "Provide a comprehensive response using only the source material above."
        ])
        
        return "\n\n".join(prompt_sections)
    
    def _post_process_response(self, response: str, sources: List[str], analysis: QueryAnalysis) -> str:
        """Post-process response based on analysis"""
        # Clean response
        response = self._clean_response(response)
        
        # Add source attribution if not already present
        if sources and not self._has_source_attribution(response):
            response += f"\n\n📚 **Sources:** {', '.join(sources)}"
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean response dynamically"""
        # Remove reasoning tokens and artifacts
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'^(Assistant|AI|Response):\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def _has_source_attribution(self, response: str) -> bool:
        """Check if response already has source attribution"""
        source_indicators = ['source:', 'sources:', '📚', 'from:', 'according to']
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in source_indicators)
    
    def _create_fallback_response(self, context: str, sources: List[str]) -> str:
        """Create fallback response"""
        if sources:
            return f"I found relevant information in {', '.join(sources)}, but encountered difficulties generating a detailed response. Please try rephrasing your question."
        else:
            return "This information is not available in my current knowledge base."
    
    def format_conversation_history(self, messages: List[ChatMessage]) -> str:
        """Format conversation history dynamically"""
        if not messages:
            return "No previous conversation."
        
        # Get recent messages based on configuration
        recent_limit = min(int(os.getenv('CONVERSATION_HISTORY_LIMIT', '4')), len(messages))
        content_limit = int(os.getenv('HISTORY_CONTENT_LIMIT', '150'))
        
        recent_messages = messages[-recent_limit:]
        history_parts = []
        
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:content_limit] + "..." if len(msg.content) > content_limit else msg.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def chat(self, query: str, session_id: str = None) -> Tuple[str, str, List[str]]:
        """Main chat function with full LLM intelligence"""
        
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
        history_limit = int(os.getenv('CONVERSATION_HISTORY_LIMIT', '6'))
        history = self.memory.get_conversation_history(session_id, limit=history_limit)
        conversation_context = self.format_conversation_history(history[:-1])
        
        # LLM-powered query analysis
        query_analysis = self.analyzer.analyze_query_with_llm(query)
        
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
        self.memory.add_message(
            session_id, assistant_message, 
            has_sources=has_relevant_info,
            query_analysis=asdict(query_analysis),
            sources_used=sources
        )
        
        # Log interaction with analysis details
        logger.info(f"💬 Session {session_id}: Intent={query_analysis.query_intent}, "
                   f"Books={len(query_analysis.detected_books)}, Sources={len(sources)}")
        
        return response, session_id, sources
    
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
                'system_ready': self.analyzer is not None
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                'total_points': 'Error',
                'collection_status': 'Unknown',
                'available_books': 0,
                'system_ready': False
            }
    
    def get_book_info_with_llm(self, book_name: str) -> Dict[str, Any]:
        """Get comprehensive book information using LLM analysis"""
        if book_name not in self.book_metadata:
            return {}
        
        metadata = self.book_metadata[book_name]
        sample_content = ' '.join(metadata.get('sample_content', []))
        
        if not sample_content:
            return {
                'name': book_name,
                'chunk_count': metadata.get('chunk_count', 0),
                'description': 'Book available in knowledge base'
            }
        
        analysis_prompt = f"""
        Analyze this philosophical work and provide a structured description:
        
        Book Title: {book_name}
        Sample Content: {sample_content[:500]}...
        
        Provide a JSON response with:
        {{
            "main_subject": "primary philosophical area/topic",
            "key_themes": ["list", "of", "main", "themes"],
            "writing_style": "description of author's style",
            "historical_period": "time period or philosophical movement",
            "brief_description": "2-3 sentence overview"
        }}
        
        Only respond with the JSON, no additional text.
        """
        
        try:
            llm_response = self._call_llm_simple(analysis_prompt)
            
            # Parse LLM analysis
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(llm_response)
            
            # Combine with metadata
            return {
                'name': book_name,
                'chunk_count': metadata.get('chunk_count', 0),
                'avg_chunk_size': metadata.get('avg_chunk_size', 0),
                'main_subject': analysis_data.get('main_subject', 'Philosophy'),
                'key_themes': analysis_data.get('key_themes', []),
                'writing_style': analysis_data.get('writing_style', 'Academic'),
                'historical_period': analysis_data.get('historical_period', 'Unknown'),
                'description': analysis_data.get('brief_description', 'Philosophical work')
            }
            
        except Exception as e:
            logger.warning(f"LLM book analysis failed for {book_name}: {e}")
            return {
                'name': book_name,
                'chunk_count': metadata.get('chunk_count', 0),
                'description': f'Philosophical work: {book_name}'
            }
    
    def adaptive_search_refinement(self, query: str, previous_results: List[SearchResult]) -> List[SearchResult]:
        """Use LLM to adaptively refine search if results are poor"""
        if not previous_results:
            return []
        
        refinement_prompt = f"""
        Original Query: "{query}"
        
        The search returned these results, but they may not be optimal:
        {self._format_results_for_llm(previous_results[:5])}
        
        Suggest 2-3 alternative search queries that might find better results for this user question.
        Focus on:
        1. Different key terms
        2. More specific concepts
        3. Alternative phrasings
        
        Respond with a JSON array of alternative queries:
        ["alternative query 1", "alternative query 2", "alternative query 3"]
        
        Only respond with the JSON array, no additional text.
        """
        
        try:
            llm_response = self._call_llm_simple(refinement_prompt)
            alternative_queries = json.loads(llm_response.strip())
            
            # Try alternative searches
            best_results = previous_results
            best_score = sum(r.score for r in previous_results[:3])
            
            for alt_query in alternative_queries:
                if isinstance(alt_query, str):
                    alt_embedding = self.embedding_model.encode(alt_query).tolist()
                    alt_results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=alt_embedding,
                        limit=10,
                        with_payload=True
                    )
                    
                    alt_processed = self._process_search_results(alt_results, QueryAnalysis(
                        query=alt_query, detected_books=[], query_intent='general',
                        complexity_level='moderate', requires_quotes=False,
                        expected_sources=3, search_strategy='broad',
                        context_requirements={}
                    ))
                    
                    alt_score = sum(r.score for r in alt_processed[:3])
                    if alt_score > best_score:
                        best_results = alt_processed
                        best_score = alt_score
            
            return best_results
            
        except Exception as e:
            logger.warning(f"Adaptive refinement failed: {e}")
            return previous_results
    
    def get_query_suggestions(self, current_books: List[str] = None) -> List[str]:
        """Generate query suggestions using LLM and available books"""
        books_context = current_books or self.available_books
        books_sample = books_context[:10]  # Use first 10 books for context
        
        suggestion_prompt = f"""
        Available philosophical works:
        {', '.join(books_sample)}
        
        Generate 5 interesting and diverse question suggestions that users might ask about these philosophical works.
        
        Make the questions:
        1. Specific and engaging
        2. Cover different types of philosophical inquiry
        3. Reference the actual available books
        4. Range from simple to complex
        
        Respond with a JSON array of 5 question strings:
        ["question 1", "question 2", "question 3", "question 4", "question 5"]
        
        Only respond with the JSON array, no additional text.
        """
        
        try:
            llm_response = self._call_llm_simple(suggestion_prompt)
            suggestions = json.loads(llm_response.strip())
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            logger.warning(f"Query suggestion generation failed: {e}")
            # Fallback to simple suggestions
            return [
                "What are the main themes in the available philosophical works?",
                "Compare different philosophical approaches to ethics.",
                "Explain key concepts from political philosophy.",
                "What arguments are made about human nature?",
                "How do these philosophers view the role of government?"
            ]

# Factory function for easy integration
def create_intelligent_philosophy_rag(qdrant_url: str, qdrant_api_key: str, 
                                    collection_name: str = "psychology_books_kb",
                                    embedding_model: str = "all-MiniLM-L6-v2",
                                    llama_model: str = "llama3.2:3b") -> IntelligentPhilosophyRAG:
    """Create an intelligent LLM-powered philosophy RAG system"""
    return IntelligentPhilosophyRAG(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llama_model=llama_model
    )

