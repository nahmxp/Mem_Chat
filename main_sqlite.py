# Standard library imports
import hashlib
import json
import os
import pickle
import sqlite3
import traceback
from datetime import datetime

# Third-party imports
import numpy as np
import streamlit as st
from cryptography.fernet import Fernet
from PyPDF2 import PdfReader

# LangChain imports
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings

# Try to import sqlite-vec, fallback to manual implementation
try:
    import sqlite_vec
    VSS_AVAILABLE = True
except ImportError:
    VSS_AVAILABLE = False
    st.warning("sqlite-vec not found. Install with: pip install sqlite-vec")

class AdvancedEncryptedSQLiteVectorStore:
    def __init__(self, storage_dir="encrypted_sqlite_storage"):
        self.storage_dir = storage_dir
        self.key_file = os.path.join(storage_dir, "key.key")
        self.db_file = os.path.join(storage_dir, "vectors.db")
        self.metadata_file = os.path.join(storage_dir, "metadata.encrypted")
        self.document_versions_file = os.path.join(storage_dir, "doc_versions.encrypted")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load or create encryption key
        self.cipher = self._get_or_create_cipher()
        
        # Initialize embeddings and detect dimension
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.embedding_dim = self._detect_embedding_dimension()
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self._setup_database()
        
        # Track document versions and metadata
        self.document_versions = self._load_document_versions()
        self.chunk_metadata = self._load_chunk_metadata()
        
    def _get_or_create_cipher(self):
        """Get existing encryption key or create a new one."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        return Fernet(key)
    
    def _detect_embedding_dimension(self):
        """Detect the actual embedding dimension from the model."""
        try:
            test_embedding = self.embeddings.embed_query("test")
            dim = len(test_embedding)
            print(f"Detected embedding dimension: {dim}")
            return dim
        except Exception as e:
            print(f"Could not detect embedding dimension: {e}")
            return 768  # Default fallback
    
    def _setup_database(self):
        """Setup SQLite database with vector support."""
        global VSS_AVAILABLE
        
        # Create regular table for document content (always needed)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS document_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                content TEXT,
                doc_name TEXT,
                chunk_index INTEGER,
                timestamp TEXT,
                hash TEXT,
                embedding BLOB
            )
        """)
        
        if VSS_AVAILABLE:
            # Enable sqlite-vec extension
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                
                # Create virtual table for vector search using detected dimension
                self.conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                        embedding float[{self.embedding_dim}]
                    )
                """)
                st.info(f"✅ SQLite-Vec extension loaded successfully (dimension: {self.embedding_dim})")
            except Exception as e:
                st.warning(f"Vector extension setup failed: {e}. Using manual vector search.")
                VSS_AVAILABLE = False
        
        self.conn.commit()
    
    def _load_document_versions(self):
        """Load document version tracking data."""
        if os.path.exists(self.document_versions_file):
            try:
                with open(self.document_versions_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                return pickle.loads(decrypted_data)
            except Exception as e:
                st.warning(f"Could not load document versions: {e}")
        return {}
    
    def _load_chunk_metadata(self):
        """Load metadata about text chunks and their source documents."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                return pickle.loads(decrypted_data)
            except Exception as e:
                st.warning(f"Could not load chunk metadata: {e}")
        return {}
    
    def _save_document_versions(self):
        """Save document version tracking data."""
        try:
            serialized_data = pickle.dumps(self.document_versions)
            encrypted_data = self.cipher.encrypt(serialized_data)
            with open(self.document_versions_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            st.error(f"Error saving document versions: {e}")
    
    def _save_chunk_metadata(self):
        """Save chunk metadata."""
        try:
            serialized_data = pickle.dumps(self.chunk_metadata)
            encrypted_data = self.cipher.encrypt(serialized_data)
            with open(self.metadata_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            st.error(f"Error saving chunk metadata: {e}")
    
    def _get_document_hash(self, text):
        """Generate a hash for document content."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_text_chunks_with_metadata(self, text, doc_name):
        """Split text into chunks and create metadata for each chunk."""
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = splitter.split_text(text)
            
            # Create metadata for each chunk
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                chunk_data.append({
                    'text': chunk,
                    'hash': chunk_hash,
                    'doc_name': doc_name,
                    'chunk_index': i,
                    'timestamp': datetime.now().isoformat()
                })
            return chunk_data
        except Exception as e:
            st.error(f"Error creating chunks: {e}")
            return []
    
    def _detect_document_changes(self, doc_name, current_text):
        """Detect if document has changed and identify new/modified chunks."""
        current_hash = self._get_document_hash(current_text)
        
        if doc_name not in self.document_versions:
            # Completely new document
            return "new", None, current_hash
        
        stored_hash = self.document_versions[doc_name]['content_hash']
        
        if stored_hash == current_hash:
            # Document unchanged
            return "unchanged", None, current_hash
        else:
            # Document updated - need to find what changed
            return "updated", stored_hash, current_hash
    
    def _remove_old_document_chunks(self, doc_name):
        """Remove chunks belonging to old version of document from SQLite."""
        try:
            # Remove from document_vectors table
            self.conn.execute("DELETE FROM document_vectors WHERE doc_name = ?", (doc_name,))
            
            if VSS_AVAILABLE:
                # Remove from VSS table (more complex, need to rebuild)
                # For now, we'll mark them as deleted in metadata
                pass
            
            # Remove from chunk metadata
            to_remove = []
            for chunk_id, chunk_meta in self.chunk_metadata.items():
                if chunk_meta.get('doc_name') == doc_name:
                    to_remove.append(chunk_id)
            
            for chunk_id in to_remove:
                del self.chunk_metadata[chunk_id]
            
            self.conn.commit()
            
        except Exception as e:
            st.error(f"Error removing old document chunks: {e}")
    
    def _add_chunks_to_db(self, chunks):
        """Add chunks to SQLite database with embeddings."""
        try:
            for chunk in chunks:
                chunk_id = f"{chunk['doc_name']}_{chunk['chunk_index']}_{chunk['hash'][:8]}"
                
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk['text'])
                
                # Insert into regular table
                self.conn.execute("""
                    INSERT OR REPLACE INTO document_vectors 
                    (chunk_id, content, doc_name, chunk_index, timestamp, hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, chunk['text'], chunk['doc_name'], 
                      chunk['chunk_index'], chunk['timestamp'], chunk['hash']))
                
                if VSS_AVAILABLE:
                    # Insert into vector table with proper format
                    try:
                        # Get the row ID from the document_vectors table
                        cursor = self.conn.execute("SELECT id FROM document_vectors WHERE chunk_id = ?", (chunk_id,))
                        row_id = cursor.fetchone()[0]
                        
                        # Convert embedding to proper format for sqlite-vec
                        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                        
                        self.conn.execute("""
                            INSERT OR REPLACE INTO vec_embeddings(rowid, embedding)
                            VALUES (?, ?)
                        """, (row_id, embedding_bytes))
                    except Exception as e:
                        st.warning(f"Vector insert warning: {e}")
                        # Fallback: store embedding as blob in main table
                        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                        self.conn.execute("""
                            UPDATE document_vectors SET embedding = ? WHERE chunk_id = ?
                        """, (embedding_blob, chunk_id))
                else:
                    # Manual embedding storage
                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                    self.conn.execute("""
                        UPDATE document_vectors SET embedding = ? WHERE chunk_id = ?
                    """, (embedding_blob, chunk_id))
                
                # Update metadata
                self.chunk_metadata[chunk_id] = chunk
            
            self.conn.commit()
            
        except Exception as e:
            st.error(f"Error adding chunks to database: {e}")
            traceback.print_exc()
    
    def similarity_search(self, query, k=8):
        """Perform similarity search using SQLite+Vec."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            if VSS_AVAILABLE:
                # Use sqlite-vec for similarity search
                try:
                    # Convert query embedding to proper format
                    query_embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
                    
                    # Use the vector search with proper SQL syntax for sqlite-vec
                    results = self.conn.execute("""
                        SELECT d.content, d.doc_name, d.chunk_index, d.chunk_id
                        FROM document_vectors d
                        JOIN vec_embeddings v ON v.rowid = d.id
                        ORDER BY vec_distance_cosine(v.embedding, ?)
                        LIMIT ?
                    """, (query_embedding_bytes, k)).fetchall()
                    
                    documents = []
                    for content, doc_name, chunk_index, chunk_id in results:
                        documents.append(Document(
                            page_content=content, 
                            metadata={
                                'chunk_id': chunk_id,
                                'doc_name': doc_name,
                                'chunk_index': chunk_index
                            }
                        ))
                    
                    return documents if documents else self._manual_similarity_search(query_embedding, k)
                    
                except Exception as e:
                    st.warning(f"Vector search failed, falling back to manual: {e}")
                    return self._manual_similarity_search(query_embedding, k)
            else:
                return self._manual_similarity_search(query_embedding, k)
            
        except Exception as e:
            st.error(f"Error in similarity search: {e}")
            traceback.print_exc()
            return []
    
    def _manual_similarity_search(self, query_embedding, k=8):
        """Manual similarity search fallback."""
        try:
            # Manual similarity search fallback
            cursor = self.conn.execute("SELECT chunk_id, content, doc_name, chunk_index, embedding FROM document_vectors WHERE embedding IS NOT NULL")
            results = []
            
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            for row in cursor:
                chunk_id, content, doc_name, chunk_index, embedding_blob = row
                
                if embedding_blob:
                    doc_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    results.append({
                        'content': content,
                        'similarity': similarity,
                        'metadata': {
                            'chunk_id': chunk_id,
                            'doc_name': doc_name,
                            'chunk_index': chunk_index
                        }
                    })
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            documents = []
            
            for result in results[:k]:
                documents.append(Document(
                    page_content=result['content'],
                    metadata=result['metadata']
                ))
            
            return documents
            
        except Exception as e:
            st.error(f"Error in manual similarity search: {e}")
            return []
    
    def process_documents(self, pdf_docs):
        """Process documents with intelligent update detection."""
        try:
            total_new_chunks = 0
            updated_documents = []
            
            for pdf in pdf_docs:
                # Extract text from PDF
                pdf_text = self._extract_pdf_text(pdf)
                if not pdf_text:
                    continue
                
                doc_name = pdf.name
                
                # Detect document changes
                change_type, old_hash, new_hash = self._detect_document_changes(doc_name, pdf_text)
                
                if change_type == "unchanged":
                    st.info(f"📄 {doc_name}: No changes detected")
                    continue
                
                elif change_type == "new":
                    st.info(f"📄 {doc_name}: New document detected")
                    new_chunks = self._get_text_chunks_with_metadata(pdf_text, doc_name)
                    
                elif change_type == "updated":
                    st.info(f"📄 {doc_name}: Document updated - processing changes")
                    
                    # Remove old chunks from SQLite
                    self._remove_old_document_chunks(doc_name)
                    
                    # Get new chunks from updated document
                    new_chunks = self._get_text_chunks_with_metadata(pdf_text, doc_name)
                    updated_documents.append(doc_name)
                
                if new_chunks:
                    # Add chunks to database
                    self._add_chunks_to_db(new_chunks)
                    
                    # Update document version tracking
                    self.document_versions[doc_name] = {
                        'content_hash': new_hash,
                        'last_updated': datetime.now().isoformat(),
                        'chunk_count': len(new_chunks)
                    }
                    
                    total_new_chunks += len(new_chunks)
            
            # Save everything if we have updates
            if total_new_chunks > 0:
                self._save_chunk_metadata()
                self._save_document_versions()
                
                st.success(f"✅ Processing complete! Added {total_new_chunks} new chunks.")
                if updated_documents:
                    st.info(f"📝 Updated documents: {', '.join(updated_documents)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            traceback.print_exc()
            return False
    
    def _extract_pdf_text(self, pdf):
        """Extract text from a single PDF file."""
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text.replace("\n", " ").strip()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
            return ""
    
    def get_stats(self):
        """Get statistics about the knowledge base."""
        total_docs = len(self.document_versions)
        total_chunks = len(self.chunk_metadata)
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'documents': list(self.document_versions.keys())
        }
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

# Global vector store instance
vector_store_manager = AdvancedEncryptedSQLiteVectorStore()

def get_conversational_chain():
    """Creates a conversational chain using the Qwen2.5 model in Ollama."""
    try:
        prompt_template = """
        Context:
        {context}
Question: {question}

        Provide a concise and accurate response based on the context.
        If no relevant answer is found, state that clearly.
        """
        model = Ollama(model="qwen2.5:7b")
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        traceback.print_exc()
        return None

def user_input(user_question):
    """Processes user query, retrieves relevant documents, and generates a response."""
    try:
        # Perform similarity search using SQLite
        docs = vector_store_manager.similarity_search(user_question, k=8)
        
        if not docs:
            return "Sorry, no relevant answer found. Try rephrasing."

        chain = get_conversational_chain()
        if not chain:
            return "Sorry, could not initialize the AI model."
            
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "Sorry, no relevant answer found.")
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        traceback.print_exc()
        return "Sorry, something went wrong."

def clear_chat_history():
    """Resets chat history in Streamlit."""
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question!"}]

def main():
    st.set_page_config(page_title="Advanced PDF Chatbot (SQLite+VSS)", page_icon="🗄️")

    with st.sidebar:
        st.title("📂 Document Management")
        
        # Show VSS status
        if VSS_AVAILABLE:
            st.success("🚀 SQLite+Vec enabled")
        else:
            st.warning("⚠️ Using manual vector search")
        
        # Show knowledge base statistics
        stats = vector_store_manager.get_stats()
        if stats['total_documents'] > 0:
            st.info(f"📊 Knowledge Base Stats:")
            st.write(f"📄 Documents: {stats['total_documents']}")
            st.write(f"📝 Text Chunks: {stats['total_chunks']}")
            
            with st.expander("📋 Document List"):
                for doc in stats['documents']:
                    doc_info = vector_store_manager.document_versions[doc]
                    st.write(f"• {doc}")
                    st.caption(f"Last updated: {doc_info['last_updated'][:19]}")
        
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        
        if st.button("🔄 Process Documents"):
            if pdf_docs:
                with st.spinner("Analyzing documents for changes..."):
                    vector_store_manager.process_documents(pdf_docs)
            else:
                st.warning("Please upload at least one PDF.")

    st.title("Chat with Your PDFs 📖🗄️ (SQLite+Vec)")

    st.sidebar.button("🗑 Clear Chat", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        response = user_input(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()