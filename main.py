from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Dict, Any
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import cohere
import json
import functools
import gradio as gr
from gradio_ui import demo


load_dotenv()

app = FastAPI(
    title="Mini-RAG Backend with Pinecone",
    description="A high-performance RAG system with citations and scores",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    text: str
    source: str | None = "user_input"
    title: str | None = "Untitled"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    include_scores: bool = True

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: int = 10
    include_scores: bool = True

# Global connection pool for Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Optimized embeddings with better timeout handling
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    request_timeout=20,  # Increased timeout to 2 minutes
    max_retries=3,        # Increased retries
    temperature=0.0        # Deterministic embeddings
)

# Optimized text splitter with better chunking strategy for citations
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Reduced chunk size to avoid timeouts
    chunk_overlap=120, # Reduced overlap proportionally
    length_function=len,
    separators=["\n", "\n\n", ". ", " ", ""]  # More intelligent splitting
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Global Cohere client with connection pooling
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Global instance for answer service
class AnswerService:
    """Service for generating answers with citations using Google Gemini LLM"""
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,  # Low temperature for consistent, factual answers
            max_output_tokens=200,
            request_timeout=20
        )
        # System prompt for citation-aware answering
        self.system_prompt = """You are an expert AI assistant that provides accurate, well-cited answers based on retrieved documents.
IMPORTANT RULES:
1. ALWAYS use inline citations [1], [2], [3] etc. when referencing information from documents
2. ONLY use information from the provided documents - do not add external knowledge
3. If you cannot answer the question from the documents, say "I cannot answer this question based on the available information"
4. Be concise but comprehensive
5. Format citations as [1], [2], [3] etc. in the text
6. Always end your answer with a summary of the key points
Document format: Each document has a chunk_id, text content, and relevance score."""
        # Human prompt template
        self.human_prompt = """Question: {query}
Retrieved Documents:
{documents}
Please provide a comprehensive answer with inline citations [1], [2], [3] etc. that reference the specific documents above. Include the relevance scores in your analysis.
Answer:"""

    def _format_documents_for_citation(self, reranked_docs: List[Dict]) -> str:
        """Format documents for the LLM with citation information"""
        formatted_docs = []
        for i, doc in enumerate(reranked_docs):
            # Extract metadata
            metadata = doc.get("metadata", {})
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            # Format each document
            doc_text = f"[{i+1}] (Score: {score:.3f}, Chunk: {metadata.get('chunk_id', 'N/A')})\n"
            doc_text += f"Source: {metadata.get('source', 'Unknown')}\n"
            doc_text += f"Title: {metadata.get('title', 'Untitled')}\n"
            doc_text += f"Text: {text[:500]}{'...' if len(text) > 500 else ''}\n"
            formatted_docs.append(doc_text)
        return "\n---\n".join(formatted_docs)

    def _extract_citations_from_answer(self, answer: str) -> List[Dict]:
        """Extract citation information from the LLM answer"""
        citations = []
        # Find all citation markers [1], [2], etc.
        import re
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        # Remove duplicates and sort
        unique_citations = sorted(list(set(matches)), key=int)
        return [{"citation_id": int(cid), "marker": f"[{cid}]"} for cid in unique_citations]

    def _map_citations_to_sources(self, citations: List[Dict], reranked_docs: List[Dict]) -> List[Dict]:
        """Map citation markers to actual source documents"""
        mapped_citations = []
        for citation in citations:
            citation_id = citation["citation_id"]
            # Map citation ID to document index (citation_id - 1 for 0-based indexing)
            if 1 <= citation_id <= len(reranked_docs):
                doc = reranked_docs[citation_id - 1]
                metadata = doc.get("metadata", {})
                mapped_citation = {
                    **citation,
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "relevance_score": doc.get("score", 0.0),
                    "text_snippet": doc.get("text", "")[:200] + "..." if len(doc.get("text", "")) > 200 else doc.get("text", ""),
                    "word_count": metadata.get("word_count", 0),
                    "position": metadata.get("position", "N/A")
                }
                mapped_citations.append(mapped_citation)
        return mapped_citations

    async def generate_answer_with_citations(
        self, 
        query: str, 
        reranked_docs: List[Dict],
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer with citations and scores
        Args:
            query: User's question
            reranked_docs: List of reranked documents with scores
            include_scores: Whether to include relevance scores in the answer
        Returns:
            Dictionary containing answer, citations, sources, and metadata
        """
        start_time = time.time()
        if not reranked_docs:
            return {
                "answer": "I cannot answer this question as no relevant documents were found.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_used": 0,
                    "status": "no_documents"
                }
            }
        try:
            # Format documents for the LLM
            formatted_docs = self._format_documents_for_citation(reranked_docs)
            # Create the prompt
            prompt = self.human_prompt.format(
                query=query,
                documents=formatted_docs
            )
            # Generate answer using Gemini
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            answer = response.content
            # Extract citations
            citations = self._extract_citations_from_answer(answer)
            # Map citations to sources
            mapped_citations = self._map_citations_to_sources(citations, reranked_docs)
            # Prepare sources list
            sources = []
            for doc in reranked_docs:
                metadata = doc.get("metadata", {})
                source_info = {
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "relevance_score": doc.get("score", 0.0),
                    "text": doc.get("text", ""),
                    "word_count": metadata.get("word_count", 0),
                    "position": metadata.get("position", "N/A"),
                    "timestamp": metadata.get("timestamp", "N/A")
                }
                sources.append(source_info)
            # Calculate statistics
            processing_time = time.time() - start_time
            avg_score = sum(doc.get("score", 0.0) for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0.0
            metadata = {
                "processing_time": processing_time,
                "documents_used": len(reranked_docs),
                "citations_found": len(citations),
                "average_relevance_score": avg_score,
                "query_length": len(query),
                "answer_length": len(answer),
                "status": "success"
            }
            return {
                "answer": answer,
                "citations": mapped_citations,
                "sources": sources,
                "metadata": metadata,
                "query": query
            }
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": processing_time,
                    "documents_used": len(reranked_docs),
                    "error": str(e),
                    "status": "error"
                },
                "query": query
            }

    def generate_answer_with_citations_sync(
        self, 
        query: str, 
        reranked_docs: List[Dict],
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for backward compatibility"""
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, use run_until_complete
            return loop.run_until_complete(
                self.generate_answer_with_citations(query, reranked_docs, include_scores)
            )
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(
                self.generate_answer_with_citations(query, reranked_docs, include_scores)
            )

# Global instance for easy access
answer_service = AnswerService()

# Convenience functions
async def generate_answer_with_citations(query: str, reranked_docs: List[Dict]) -> Dict[str, Any]:
    """Generate answer with citations using the global answer service"""
    return await answer_service.generate_answer_with_citations(query, reranked_docs)

def generate_answer_with_citations_sync(query: str, reranked_docs: List[Dict]) -> Dict[str, Any]:
    """Synchronous version of generate_answer_with_citations"""
    return answer_service.generate_answer_with_citations_sync(query, reranked_docs)

# ---------- optimized helpers ----------
def _batch_list(items, batch_size):
    """Optimized batching with memory efficiency"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def _parallel_embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Parallel embedding with optimized batch sizes and better error handling"""
    if not texts:
        return []
    # Use smaller batches to avoid timeouts
    batches = list(_batch_list(texts, batch_size))
    async def embed_single_batch(batch):
        loop = asyncio.get_event_loop()
        max_retries = 3
        base_delay = 2
        for attempt in range(max_retries):
            try:
                print(f"Embedding batch of {len(batch)} texts (attempt {attempt + 1})")
                result = await loop.run_in_executor(executor, embeddings.embed_documents, batch)
                print(f"Successfully embedded batch of {len(batch)} texts")
                return result
            except Exception as e:
                print(f"Batch embed attempt {attempt + 1} failed: {e}")
                if "504" in str(e) or "Deadline Exceeded" in str(e):
                    print(f"Timeout error detected, retrying with smaller batch...")
                    # Split batch in half and retry
                    if len(batch) > 8:
                        half = len(batch) // 2
                        print(f"Splitting batch from {len(batch)} to {half} texts")
                        # Process both halves concurrently
                        left_task = embed_single_batch(batch[:half])
                        right_task = embed_single_batch(batch[half:])
                        left_result, right_result = await asyncio.gather(left_task, right_task)
                        return left_result + right_result
                    else:
                        # If batch is already small, wait and retry
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Waiting {delay}s before retry...")
                            await asyncio.sleep(delay)
                        else:
                            print(f"All retry attempts failed for batch of {len(batch)} texts")
                            raise e
                else:
                    # Non-timeout error, try to retry
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Non-timeout error, waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All retry attempts failed for batch of {len(batch)} texts")
                        raise e
        # If we get here, all retries failed
        raise Exception(f"Failed to embed batch after {max_retries} attempts")
    # Process batches with better error handling
    all_embeddings = []
    failed_batches = []
    for i, batch in enumerate(batches):
        try:
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} texts")
            batch_result = await embed_single_batch(batch)
            all_embeddings.extend(batch_result)
            print(f"Batch {i+1} completed successfully")
        except Exception as e:
            print(f"Batch {i+1} failed completely: {e}")
            failed_batches.append((i, batch, e))
            # Add placeholder embeddings to maintain count
            all_embeddings.extend([[0.0] * 768] * len(batch))
    if failed_batches:
        print(f"Warning: {len(failed_batches)} batches failed during embedding")
        for i, batch, error in failed_batches:
            print(f"  Batch {i+1}: {len(batch)} texts failed - {error}")
    return all_embeddings

async def _parallel_pinecone_upsert(vectors: List[Dict[str, Any]], batch_size: int = 50):
    """Parallel Pinecone upserts with optimized batching"""
    # 🔍 DEBUG: Add this code to see what's happening
    print(f"=== DEBUG: Pinecone Upsert Started ===")
    print(f"Total vectors to upsert: {len(vectors)}")
    if vectors:
        print(f"First vector ID: {vectors[0]['id']}")
        print(f"First vector embedding length: {len(vectors[0]['values'])}")
        print(f"First vector text preview: {vectors[0]['metadata']['text'][:50]}...")
    else:
        print("❌ NO VECTORS TO UPSERT - This is the problem!")
        return
    print(f"Batch size: {batch_size}")
    print("=== DEBUG: End ===")
    if not vectors:
        return
    # Reduced batch size for better reliability
    batches = list(_batch_list(vectors, batch_size))
    async def upsert_batch(batch):
        print(f"🔄 Processing batch with {len(batch)} vectors")
        loop = asyncio.get_event_loop()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Upsert attempt {attempt + 1} for batch size {len(batch)}")
                # ✅ FIXED: Use lambda function
                result = await loop.run_in_executor(executor, lambda: index.upsert(vectors=batch))
                print(f"✅ Upsert successful for {len(batch)} vectors")
                return result
            except Exception as e:
                print(f"❌ Upsert attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Retry with smaller batch
                    if len(batch) > 25:
                        half = len(batch) // 2
                        print(f"Splitting upsert batch from {len(batch)} to {half}")
                        await upsert_batch(batch[:half])
                        await upsert_batch(batch[half:])
                        return
                    else:
                        await asyncio.sleep(2 ** attempt)
                else:
                    raise
    # Process upserts concurrently
    tasks = [upsert_batch(batch) for batch in batches]
    await asyncio.gather(*tasks, return_exceptions=True)

def _create_enhanced_metadata(doc: Document, source: str, title: str, chunk_id: int, total_chunks: int) -> Dict[str, Any]:
    """Create enhanced metadata for better citations and retrieval"""
    chunk_hash = sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return {
        "source": source,
        "title": title,
        "text": doc.page_content,
        "chunk_id": chunk_id,
        "position": chunk_id,
        "section": f"chunk_{chunk_id}",
        "total_chunks": total_chunks,
        "length": len(doc.page_content),
        "hash": chunk_hash,
        "timestamp": time.time(),
        "word_count": len(doc.page_content.split()),
        "type": "text_chunk"
    }

# ---------- main optimized function ----------
async def ingest_text_async(
    text: str,
    source: str = "user_input",
    title: str = "Untitled",
    embed_batch_size: int = 50,  # Reduced for better reliability
    upsert_batch_size: int = 50  # Reduced for better reliability
) -> Dict[str, Any]:
    """
    Enhanced async ingestion with better error handling and timeout management:
    - Parallel text splitting with enhanced metadata
    - Concurrent batch embedding with retry logic
    - Parallel Pinecone upserts
    - Smart error handling and fallbacks
    - Returns detailed ingestion statistics
    """
    start_time = time.time()
    try:
        # 1) Parallel text splitting
        print(f"Starting text splitting for {len(text)} characters...")
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(executor, text_splitter.split_text, text)
        if not docs:
            return {
                "chunks_ingested": 0,
                "processing_time": 0,
                "total_words": 0,
                "chunk_stats": {},
                "status": "no_content"
            }
        total_chunks = len(docs)
        total_words = sum(len(doc.split()) for doc in docs)
        print(f"Text split into {total_chunks} chunks with {total_words} total words")
        # Create documents with enhanced metadata for citations
        documents = []
        for i, doc in enumerate(docs):
            document = Document(
                page_content=doc, 
                metadata=_create_enhanced_metadata(Document(page_content=doc), source, title, i, total_chunks)
            )
            documents.append(document)
        # 2) Parallel batch embedding with better error handling
        print(f"Starting embedding process for {total_chunks} chunks...")
        texts = [d.page_content for d in documents]
        embeddings_list = await _parallel_embed_batch(texts, embed_batch_size)
        if len(embeddings_list) != len(documents):
            print(f"Warning: Embedding count mismatch. Expected {len(documents)}, got {len(embeddings_list)}")
            # Pad embeddings if needed
            while len(embeddings_list) < len(documents):
                embeddings_list.append([0.0] * 768)
            # Truncate if too many
            embeddings_list = embeddings_list[:len(documents)]
        print(f"Embedding completed: {len(embeddings_list)} vectors generated")
        # 3) Prepare vectors for parallel upsert
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
            # vec_id = f"{source}_{title}_{doc.metadata['hash']}"
            vec_id = f"{source}_{title}_{i}_{doc.metadata['hash']}"
            vectors.append({
                "id": vec_id,
                "values": emb,
                "metadata": doc.metadata
            })
        # 4) Parallel Pinecone upsert
        print(f"Starting Pinecone upsert for {len(vectors)} vectors...")
        await _parallel_pinecone_upsert(vectors, upsert_batch_size)
        print("Pinecone upsert completed successfully")
        elapsed_time = time.time() - start_time
        # Calculate chunk statistics
        chunk_stats = {
            "avg_chunk_size": total_words / total_chunks if total_chunks > 0 else 0,
            "chunk_size_range": f"{min(len(doc.split()) for doc in docs)}-{max(len(doc.split()) for doc in docs)}",
            "overlap_percentage": "15%",
            "embedding_dimensions": len(embeddings_list[0]) if embeddings_list else 0
        }
        print(f"✅ Successfully ingested {total_chunks} chunks in {elapsed_time:.2f}s")
        print(f"   Total words: {total_words}, Avg chunk size: {chunk_stats['avg_chunk_size']:.1f} words")
        return {
            "chunks_ingested": total_chunks,
            "processing_time": elapsed_time,
            "total_words": total_words,
            "chunk_stats": chunk_stats,
            "status": "success",
            "source": source,
            "title": title,
            "chunk_ids": list(range(total_chunks))
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Ingestion failed after {elapsed_time:.2f}s: {e}")
        return {
            "chunks_ingested": 0,
            "processing_time": elapsed_time,
            "total_words": len(text.split()) if text else 0,
            "chunk_stats": {},
            "status": "error",
            "error": str(e),
            "source": source,
            "title": title
        }

# Backward compatibility
def ingest_text(
    text: str,
    source: str = "user_input",
    title: str = "Untitled",
    embed_batch_size: int = 50,
    upsert_batch_size: int = 50
) -> Dict[str, Any]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(ingest_text_async(text, source, title, embed_batch_size, upsert_batch_size))

# Simple cache for reranking results
_rerank_cache = {}
_cache_ttl = 600  # 10 minutes

def _get_cache_key(query: str, docs: List[str], top_k: int) -> str:
    """Generate cache key for reranking"""
    # Create a hash of the query and first few characters of each doc
    doc_signatures = [doc[:100] for doc in docs[:5]]  # First 100 chars of first 5 docs
    content = f"{query}_{top_k}_{'_'.join(doc_signatures)}"
    return f"rerank_{hash(content) % 10000}"

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < _cache_ttl

async def rerank_async(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """Async reranking with caching and optimized processing"""
    # Handle empty documents list
    if not docs:
        return []
    # Check cache first
    cache_key = _get_cache_key(query, [d["metadata"]["text"] for d in docs], top_k)
    if cache_key in _rerank_cache:
        cached_result, timestamp = _rerank_cache[cache_key]
        if _is_cache_valid(timestamp):
            return cached_result
    # docs: list of dicts with { "text": str, "metadata": {...} }
    rerank_docs = [d["metadata"]["text"] for d in docs]
    # Handle case where all documents might be empty strings
    if not any(rerank_docs):
        return []
    try:
        # Use thread pool for Cohere API call
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=rerank_docs,
                top_n=top_k
            )
        )
        ranked = []
        # Access the results from the response object
        for r in response.results:
            ranked.append({
                "text": rerank_docs[r.index],
                "metadata": docs[r.index]["metadata"],
                "score": r.relevance_score
            })
        # Cache the result
        _rerank_cache[cache_key] = (ranked, time.time())
        # Clean old cache entries
        if len(_rerank_cache) > 500:
            current_time = time.time()
            _rerank_cache.clear()
        return ranked
    except Exception as e:
        print(f"Rerank error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original docs if reranking fails
        fallback_result = [
            {
                "text": doc["metadata"]["text"], 
                "metadata": doc["metadata"], 
                "score": 0.0
            } 
            for doc in docs[:top_k]
        ]
        return fallback_result

def rerank(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(rerank_async(query, docs, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(rerank_async(query, docs, top_k))

# Batch reranking for multiple queries
async def batch_rerank_async(queries: List[str], docs_list: List[List[Dict]], top_k: int = 5) -> List[List[Dict]]:
    """Batch reranking for multiple queries - much faster than individual calls"""
    if not queries or not docs_list:
        return []
    # Process all reranking tasks in parallel
    rerank_tasks = [
        rerank_async(query, docs, top_k) 
        for query, docs in zip(queries, docs_list)
    ]
    results = await asyncio.gather(*rerank_tasks, return_exceptions=True)
    # Handle any failed reranking operations
    final_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Batch rerank error: {result}")
            final_results.append([])
        else:
            final_results.append(result)
    return final_results

def batch_rerank(queries: List[str], docs_list: List[List[Dict]], top_k: int = 5) -> List[List[Dict]]:
    """Synchronous wrapper for batch reranking"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(batch_rerank_async(queries, docs_list, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(batch_rerank_async(queries, docs_list, top_k))

# Simple in-memory cache for query embeddings
_embedding_cache = {}
_cache_ttl = 300  # 5 minutes

def _get_embedding_cache_key(query: str) -> str:
    """Generate cache key for query"""
    return f"embed_{hash(query) % 10000}"

def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < _cache_ttl

async def safe_embed_query_async(query: str, retries: int = 3, delay: float = 1.0) -> List[float]:
    """Async embedding with caching and retries"""
    # Check cache first
    cache_key = _get_embedding_cache_key(query)
    if cache_key in _embedding_cache:
        cached_emb, timestamp = _embedding_cache[cache_key]
        if _is_cache_valid(timestamp):
            return cached_emb
    # Not in cache or expired, fetch new embedding
    for attempt in range(retries):
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(executor, embeddings.embed_query, query)
            # Cache the result
            _embedding_cache[cache_key] = (embedding, time.time())
            # Clean old cache entries
            if len(_embedding_cache) > 1000:
                current_time = time.time()
                _embedding_cache.clear()
            return embedding
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise e

def safe_embed_query(query: str, retries: int = 3, delay: float = 1.0) -> List[float]:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_in_executor
        return loop.run_until_complete(safe_embed_query_async(query, retries, delay))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(safe_embed_query_async(query, retries, delay))

async def retrieve_async(query: str, top_k: int = 10) -> List:
    """Async retrieval with optimized query processing"""
    if len(query) > 2000:  # safeguard for Gemini
        query = query[:2000]
    # Get query embedding
    query_emb = await safe_embed_query_async(query)
    # Query Pinecone with optimized parameters
    try:
        results = index.query(
            vector=query_emb, 
            top_k=top_k, 
            include_metadata=True,
            include_values=False  # Don't return vectors to save bandwidth
        )
        return results.matches
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

def retrieve(query: str, top_k: int = 10) -> List:
    """Synchronous wrapper for backward compatibility"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, use run_until_complete
        return loop.run_until_complete(retrieve_async(query, top_k))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(retrieve_async(query, top_k))

# Batch retrieval for multiple queries
async def batch_retrieve_async(queries: List[str], top_k: int = 10) -> List[List]:
    """Batch retrieval for multiple queries - much faster than individual calls"""
    if not queries:
        return []
    # Get embeddings for all queries in parallel
    embedding_tasks = [safe_embed_query_async(query) for query in queries]
    embeddings_list = await asyncio.gather(*embedding_tasks, return_exceptions=True)
    # Filter out failed embeddings
    valid_embeddings = []
    valid_queries = []
    for i, emb in enumerate(embeddings_list):
        if not isinstance(emb, Exception):
            valid_embeddings.append(emb)
            valid_queries.append(queries[i])
    if not valid_embeddings:
        return [[] for _ in queries]
    # Batch query Pinecone (if supported) or parallel individual queries
    try:
        # Try batch query if available
        results = index.query(
            vectors=valid_embeddings,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return results.matches
    except:
        # Fallback to parallel individual queries
        query_tasks = [retrieve_async(query, top_k) for query in valid_queries]
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        # Pad results for failed queries
        final_results = []
        query_idx = 0
        for i in range(len(queries)):
            if query_idx < len(valid_queries) and queries[i] == valid_queries[query_idx]:
                if isinstance(results[query_idx], Exception):
                    final_results.append([])
                else:
                    final_results.append(results[query_idx])
                query_idx += 1
            else:
                final_results.append([])
        return final_results

# Performance monitoring
class PerformanceMetrics:
    """Track performance metrics for optimization"""
    def __init__(self):
        self.ingestion_times = []
        self.query_times = []
        self.embedding_times = []
        self.pinecone_times = []
        self.timeout_errors = 0
        self.retry_attempts = 0

    def add_ingestion_time(self, time_taken: float):
        self.ingestion_times.append(time_taken)
        if len(self.ingestion_times) > 100:
            self.ingestion_times.pop(0)

    def add_query_time(self, time_taken: float):
        self.query_times.append(time_taken)
        if len(self.query_times) > 100:
            self.query_times.pop(0)

    def add_timeout_error(self):
        self.timeout_errors += 1

    def add_retry_attempt(self):
        self.retry_attempts += 1

    def get_average_ingestion_time(self) -> float:
        return sum(self.ingestion_times) / len(self.ingestion_times) if self.ingestion_times else 0

    def get_average_query_time(self) -> float:
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0

    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability-focused metrics"""
        return {
            "timeout_errors": self.timeout_errors,
            "retry_attempts": self.retry_attempts,
            "success_rate": max(0, 1 - (self.timeout_errors / max(1, len(self.ingestion_times) + len(self.query_times))))
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            "avg_ingestion_time": self.get_average_ingestion_time(),
            "avg_query_time": self.get_average_query_time(),
            "total_ingestions": len(self.ingestion_times),
            "total_queries": len(self.query_times),
            "reliability": self.get_reliability_metrics()
        }

# Global performance tracker
performance_metrics = PerformanceMetrics()

@app.post("/ingest")
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """Ingest text with enhanced error handling and timeout prevention"""
    try:
        print(f"🚀 Starting ingestion for text of {len(request.text)} characters")
        # Use configuration-optimized settings
        result = await ingest_text_async(
            text=request.text,
            source=request.source,
            title=request.title,
            embed_batch_size=32,  # Reduced for reliability
            upsert_batch_size=50  # Reduced for reliability
        )
        if result['status'] == 'success':
            print(f"✅ Ingestion successful: {result['chunks_ingested']} chunks")
            return {
                "status": "success",
                "message": f"Successfully ingested {result['chunks_ingested']} chunks",
                "details": result
            }
        else:
            print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            return {
                "status": "error",
                "message": f"Ingestion failed: {result.get('error', 'Unknown error')}",
                "details": result
            }
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        print(f"❌ {error_msg}")
        # Check if it's a timeout error
        if "504" in str(e) or "Deadline Exceeded" in str(e):
            return {
                "status": "error",
                "message": "Request timed out. Please try with a smaller text or check your connection.",
                "error_type": "timeout",
                "suggestion": "Consider breaking your text into smaller sections."
            }
        return {
            "status": "error",
            "message": error_msg,
            "error_type": "general"
        }

@app.post("/ingest/sync")
async def ingest_sync(req: IngestRequest):
    """Synchronous ingestion for backward compatibility"""
    try:
        result = ingest_text(req.text, req.source, req.title)
        return {
            "status": "success",
            "data": result,
            "message": f"Successfully ingested {result.get('chunks_ingested', 0)} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
async def query_docs(req: QueryRequest):
    """Complete RAG pipeline: retrieve → rerank → answer with citations"""
    start_time = time.time()
    try:
        # Step 1: Retrieve relevant documents
        print(f"🔍 Retrieving documents for query: {req.query}")
        matches = await retrieve_async(req.query, top_k=req.top_k)
        if not matches:
            return {
                "status": "no_results",
                "answer": "No relevant documents found for your query.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": 0,
                    "status": "no_documents"
                },
                "query": req.query
            }
        # Step 2: Prepare documents for reranking
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)  # Pinecone similarity score
            }
            docs.append(doc)
        print(f"📚 Retrieved {len(docs)} documents, proceeding to reranking...")
        # Step 3: Rerank documents for better relevance
        reranked = await rerank_async(req.query, docs, top_k=min(req.top_k, 5))
        if not reranked:
            print("⚠️ Reranking failed, using original documents")
            reranked = docs[:5]  # Fallback to top 5 original docs
        print(f"🎯 Reranked to {len(reranked)} documents, generating answer...")
        # Step 4: Generate answer with citations and scores
        answer_result = await generate_answer_with_citations(req.query, reranked)
        # Add overall processing time
        total_time = time.time() - start_time
        answer_result["metadata"]["total_processing_time"] = total_time
        answer_result["metadata"]["retrieval_time"] = total_time - answer_result["metadata"].get("processing_time", 0)
        print(f"✅ Answer generated in {total_time:.2f}s")
        return {
            "status": "success",
            **answer_result,
            "request_time": total_time
        }
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Query processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/query/sync")
async def query_docs_sync(req: QueryRequest):
    """Synchronous query endpoint for backward compatibility"""
    start_time = time.time()
    try:
        # Step 1: Retrieve using sync function
        matches = retrieve(req.query, top_k=req.top_k)
        if not matches:
            return {
                "status": "no_results",
                "answer": "No relevant documents found for your query.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "documents_retrieved": 0,
                    "status": "no_documents"
                },
                "query": req.query
            }
        # Step 2: Prepare documents
        docs = []
        for match in matches:
            doc = {
                "metadata": match.metadata,
                "text": match.metadata.get("text", ""),
                "score": getattr(match, 'score', 0.0)
            }
            docs.append(doc)
        # Step 3: Rerank using sync function
        reranked = rerank(req.query, docs, top_k=min(req.top_k, 5))
        if not reranked:
            reranked = docs[:5]
        # Step 4: Generate answer with citations
        answer_result = generate_answer_with_citations_sync(req.query, reranked)
        total_time = time.time() - start_time
        answer_result["metadata"]["total_processing_time"] = total_time
        return {
            "status": "success",
            **answer_result,
            "request_time": total_time
        }
    except Exception as e:
        total_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/batch-query")
async def batch_query_docs(req: BatchQueryRequest):
    """Batch query processing for multiple questions"""
    start_time = time.time()
    try:
        results = []
        for query in req.queries:
            # Process each query individually
            query_result = await query_docs(QueryRequest(
                query=query,
                top_k=req.top_k,
                include_scores=req.include_scores
            ))
            results.append(query_result)
        total_time = time.time() - start_time
        return {
            "status": "success",
            "results": results,
            "total_queries": len(req.queries),
            "total_processing_time": total_time,
            "average_time_per_query": total_time / len(req.queries) if req.queries else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch query processing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Mini-RAG Backend",
        "version": "1.0.0",
        "features": [
            "async_ingestion",
            "vector_retrieval", 
            "document_reranking",
            "llm_answering",
            "citation_system",
            "score_tracking"
        ]
    }

@app.get("/performance")
async def performance_metrics_endpoint():
    """Get performance metrics"""
    return {
        "status": "success",
        "metrics": performance_metrics.get_performance_summary(),
        "system_info": {
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "embedding_dimensions": 768,
            "vector_db": "Pinecone",
            "llm_provider": "Google Gemini Pro",
            "reranker": "Cohere"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get index stats from Pinecone
        stats = index.describe_index_stats()
        return {
            "status": "success",
            "vector_db_stats": {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": stats.index_name,
                "metric": stats.metric
            },
            "system_config": {
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "embedding_model": "models/embedding-001",
                "llm_model": "gemini-1.5-flash",
                "reranker_model": "rerank-english-v3.0"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get stats: {str(e)}"
        )
    
# Mount Gradio app inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")