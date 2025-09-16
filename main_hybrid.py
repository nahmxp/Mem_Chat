"""
Enhanced main application with both RAG and Fine-tuned model options.
This gives you the choice between vector retrieval (RAG) and direct 
fine-tuned model inference for answering questions.
"""

# Standard library imports
import hashlib
import json
import os
import pickle
import traceback
from datetime import datetime

# Third-party imports
import streamlit as st
from cryptography.fernet import Fernet
from PyPDF2 import PdfReader

# LangChain imports
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Import the fine-tuned model inference class
try:
    from test_finetuned_model import FineTunedModelInference
    FINETUNED_MODEL_AVAILABLE = True
except ImportError:
    FINETUNED_MODEL_AVAILABLE = False

# Import the original AdvancedEncryptedVectorStore class
from main import AdvancedEncryptedVectorStore, get_conversational_chain

class HybridChatApp:
    def __init__(self):
        self.vector_store_manager = AdvancedEncryptedVectorStore()
        self.finetuned_model = None
        
        # Initialize fine-tuned model if available
        if FINETUNED_MODEL_AVAILABLE:
            self.load_finetuned_model()
    
    def load_finetuned_model(self):
        """Load the fine-tuned model if it exists."""
        try:
            if os.path.exists("./finetuned_model"):
                self.finetuned_model = FineTunedModelInference("./finetuned_model")
                return True
        except Exception as e:
            st.warning(f"Could not load fine-tuned model: {e}")
        return False
    
    def rag_response(self, user_question):
        """Generate response using RAG (vector retrieval) approach."""
        try:
            # Load the current vector store
            current_store = self.vector_store_manager.load_existing_index()
            
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
            st.error(f"Error generating RAG response: {e}")
            return "Sorry, something went wrong with the RAG approach."
    
    def finetuned_response(self, user_question):
        """Generate response using fine-tuned model."""
        if not self.finetuned_model:
            return "Fine-tuned model not available. Please train a model first using train_conversational.py"
        
        try:
            response = self.finetuned_model.generate_response(user_question)
            return response
        except Exception as e:
            st.error(f"Error generating fine-tuned response: {e}")
            return "Sorry, something went wrong with the fine-tuned model."
    
    def run_app(self):
        """Main Streamlit application."""
        st.set_page_config(page_title="Hybrid PDF Chatbot", page_icon="🔀")

        with st.sidebar:
            st.title("📂 Document Management")
            
            # Show knowledge base statistics
            stats = self.vector_store_manager.get_stats()
            if stats['total_documents'] > 0:
                st.info(f"📊 Knowledge Base Stats:")
                st.write(f"📄 Documents: {stats['total_documents']}")
                st.write(f"📝 Text Chunks: {stats['total_chunks']}")
                
                with st.expander("📋 Document List"):
                    for doc in stats['documents']:
                        doc_info = self.vector_store_manager.document_versions[doc]
                        st.write(f"• {doc}")
                        st.caption(f"Last updated: {doc_info['last_updated'][:19]}")
            
            st.subheader("Upload PDFs")
            pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
            
            if st.button("🔄 Process Documents"):
                if pdf_docs:
                    with st.spinner("Analyzing documents for changes..."):
                        self.vector_store_manager.process_documents(pdf_docs)
                else:
                    st.warning("Please upload at least one PDF.")
            
            st.divider()
            
            # Model selection
            st.subheader("🤖 AI Model Selection")
            
            model_options = ["RAG (Vector Retrieval)"]
            if self.finetuned_model:
                model_options.append("Fine-tuned Model")
            
            selected_model = st.radio("Choose AI approach:", model_options)
            
            if selected_model == "RAG (Vector Retrieval)":
                st.info("🔍 Uses vector similarity search to find relevant document chunks, then generates answers.")
                st.session_state.model_mode = "rag"
            else:
                st.info("🧠 Uses a model fine-tuned on your document content to generate answers directly.")
                st.session_state.model_mode = "finetuned"
            
            st.divider()
            
            # Training section
            st.subheader("🎯 Model Training")
            if st.button("📊 Extract Training Data"):
                with st.spinner("Extracting data from FAISS storage..."):
                    try:
                        from extract_faiss_data import FAISSDataExtractor
                        extractor = FAISSDataExtractor()
                        dataset_path = extractor.create_training_dataset()
                        if dataset_path:
                            st.success(f"Training data created: {dataset_path}")
                        else:
                            st.error("No training data could be extracted. Please upload documents first.")
                    except Exception as e:
                        st.error(f"Error extracting training data: {e}")
            
            if st.button("🚀 Start Fine-tuning"):
                st.info("Fine-tuning started! Check the terminal for progress. This may take a while...")
                st.code("python train_conversational.py", language="bash")

        # Main chat interface
        st.title("🔀 Hybrid PDF Chatbot")
        
        # Show current model mode
        if hasattr(st.session_state, 'model_mode'):
            if st.session_state.model_mode == "rag":
                st.info("🔍 **Current Mode**: RAG (Retrieval-Augmented Generation)")
            else:
                st.info("🧠 **Current Mode**: Fine-tuned Model")

        st.sidebar.button("🗑 Clear Chat", on_click=self.clear_chat_history)

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question! You can choose between RAG or fine-tuned model approaches."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask me anything..."):
            # Determine which model to use
            model_mode = getattr(st.session_state, 'model_mode', 'rag')
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response based on selected model
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    if model_mode == "finetuned" and self.finetuned_model:
                        response = self.finetuned_response(prompt)
                        response = f"🧠 **[Fine-tuned Model]**\n\n{response}"
                    else:
                        response = self.rag_response(prompt)
                        response = f"🔍 **[RAG Model]**\n\n{response}"
                
                st.write(response)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def clear_chat_history(self):
        """Clear chat history."""
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question! You can choose between RAG or fine-tuned model approaches."}]

if __name__ == "__main__":
    app = HybridChatApp()
    app.run_app()