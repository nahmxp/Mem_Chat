import streamlit as st
import traceback
import hashlib
import pickle
import os
import json
from datetime import datetime
from cryptography.fernet import Fernet
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

class AdvancedEncryptedVectorStore:
    def __init__(self, storage_dir="encrypted_faiss_storage"):
        self.storage_dir = storage_dir
        self.key_file = os.path.join(storage_dir, "key.key")
        self.index_file = os.path.join(storage_dir, "faiss_index.encrypted")
        self.metadata_file = os.path.join(storage_dir, "metadata.encrypted")
        self.document_versions_file = os.path.join(storage_dir, "doc_versions.encrypted")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load or create encryption key
        self.cipher = self._get_or_create_cipher()
        
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
    
    def _get_new_chunks_from_updated_document(self, doc_name, current_text):
        """Extract only new chunks from an updated document."""
        # Get current chunks
        current_chunks = self._get_text_chunks_with_metadata(current_text, doc_name)
        
        # Get previously stored chunk hashes for this document
        old_chunk_hashes = set()
        for chunk_id, chunk_meta in self.chunk_metadata.items():
            if chunk_meta.get('doc_name') == doc_name:
                old_chunk_hashes.add(chunk_meta['hash'])
        
        # Find new chunks (chunks not in old_chunk_hashes)
        new_chunks = []
        for chunk_data in current_chunks:
            if chunk_data['hash'] not in old_chunk_hashes:
                new_chunks.append(chunk_data)
        
        return new_chunks
    
    def _remove_old_document_chunks(self, doc_name, vector_store):
        """Remove chunks belonging to old version of document from vector store."""
        try:
            # This is complex with FAISS as it doesn't support deletion by metadata
            # We'll rebuild the index without the old document's chunks
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # Collect all chunks except those from the updated document
            remaining_chunks = []
            remaining_metadata = {}
            
            for chunk_id, chunk_meta in self.chunk_metadata.items():
                if chunk_meta.get('doc_name') != doc_name:
                    remaining_chunks.append(chunk_meta['text'])
                    remaining_metadata[chunk_id] = chunk_meta
            
            # Update chunk metadata to remove old document chunks
            self.chunk_metadata = remaining_metadata
            
            # Create new vector store with remaining chunks
            if remaining_chunks:
                new_vector_store = FAISS.from_texts(remaining_chunks, embedding=embeddings)
                return new_vector_store
            else:
                return None
                
        except Exception as e:
            st.error(f"Error removing old document chunks: {e}")
            return vector_store
    
    def load_existing_index(self):
        """Load existing FAISS index if available."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                
                # Split the combined data back into FAISS components
                separator = b"<!SEPARATOR!>"
                parts = decrypted_data.split(separator)
                
                if len(parts) != 2:
                    st.warning("Invalid encrypted index format")
                    return None
                
                faiss_data, pkl_data = parts
                
                # Save temporarily to load with FAISS
                temp_dir = "temp_faiss_load"
                os.makedirs(temp_dir, exist_ok=True)
                
                with open(os.path.join(temp_dir, "index.faiss"), 'wb') as f:
                    f.write(faiss_data)
                with open(os.path.join(temp_dir, "index.pkl"), 'wb') as f:
                    f.write(pkl_data)
                
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
                
                # Clean up temp files
                import shutil
                shutil.rmtree(temp_dir)
                
                return vector_store
            except Exception as e:
                st.warning(f"Could not load existing index: {e}")
        return None
    
    def save_index(self, vector_store):
        """Save FAISS index with encryption."""
        try:
            # Save to temporary location first
            temp_dir = "temp_faiss_save"
            os.makedirs(temp_dir, exist_ok=True)
            vector_store.save_local(temp_dir)
            
            # Read the saved files
            with open(os.path.join(temp_dir, "index.faiss"), 'rb') as f:
                faiss_data = f.read()
            with open(os.path.join(temp_dir, "index.pkl"), 'rb') as f:
                pkl_data = f.read()
            
            # Combine with separator and encrypt
            separator = b"<!SEPARATOR!>"
            combined_data = faiss_data + separator + pkl_data
            encrypted_data = self.cipher.encrypt(combined_data)
            
            # Save encrypted data
            with open(self.index_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            st.error(f"Error saving encrypted index: {e}")
    
    def process_documents(self, pdf_docs):
        """Process documents with intelligent update detection."""
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = self.load_existing_index()
            
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
                    
                    # Remove old chunks from vector store
                    if vector_store:
                        vector_store = self._remove_old_document_chunks(doc_name, vector_store)
                    
                    # Get new chunks from updated document
                    new_chunks = self._get_text_chunks_with_metadata(pdf_text, doc_name)
                    updated_documents.append(doc_name)
                
                if new_chunks:
                    # Extract just the text for vector store
                    chunk_texts = [chunk['text'] for chunk in new_chunks]
                    
                    # Add to vector store
                    if vector_store and chunk_texts:
                        vector_store.add_texts(chunk_texts)
                    elif chunk_texts:
                        vector_store = FAISS.from_texts(chunk_texts, embedding=embeddings)
                    
                    # Update metadata
                    for chunk in new_chunks:
                        chunk_id = f"{doc_name}_{chunk['chunk_index']}_{chunk['hash'][:8]}"
                        self.chunk_metadata[chunk_id] = chunk
                    
                    # Update document version tracking
                    self.document_versions[doc_name] = {
                        'content_hash': new_hash,
                        'last_updated': datetime.now().isoformat(),
                        'chunk_count': len(new_chunks)
                    }
                    
                    total_new_chunks += len(new_chunks)
            
            # Save everything if we have updates
            if total_new_chunks > 0:
                self.save_index(vector_store)
                self._save_chunk_metadata()
                self._save_document_versions()
                
                st.success(f"✅ Processing complete! Added {total_new_chunks} new chunks.")
                if updated_documents:
                    st.info(f"📝 Updated documents: {', '.join(updated_documents)}")
            
            return vector_store
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            traceback.print_exc()
            return None
    
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

# Global vector store instance
vector_store_manager = AdvancedEncryptedVectorStore()

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
        # Load the current vector store (includes all historical data)
        current_store = vector_store_manager.load_existing_index()
        
        if not current_store:
            return "No knowledge base available. Please upload some documents first."
        
        docs = current_store.similarity_search(user_question, k=8)
        
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
    st.set_page_config(page_title="Advanced PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("📂 Document Management")
        
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

    st.title("Chat with Your PDFs 📖🤖 (Smart Updates)")

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