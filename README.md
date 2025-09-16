# Mem_Chat - Advanced PDF Chatbot with Smart Updates 🤖📖

An intelligent PDF document chatbot that features smart incremental updates, encrypted storage, and local AI processing using Ollama.

## 🎯 Project Overview

Mem_Chat is an advanced PDF chatbot application that allows users to upload PDF documents and have intelligent conversations with their content. The key innovation is its **smart update system** that only processes new or changed content when documents are re-uploaded, making it highly efficient for large document collections.

## ✨ Key Features

### 🔒 **Encrypted Vector Storage**
- Uses FAISS vector database with military-grade encryption
- All document embeddings and metadata are encrypted at rest
- Secure key management with automatic key generation

### 🧠 **Smart Document Updates**
- **Incremental Processing**: Only processes new or changed content
- **Version Tracking**: Maintains document versions and change history
- **Hash-based Detection**: Uses SHA256 hashing to detect document changes
- **Chunk-level Updates**: Identifies and updates only modified text chunks

### 💬 **Interactive Chat Interface**
- Clean, modern Streamlit web interface
- Real-time conversation with your documents
- Chat history preservation
- Document management sidebar

### 📊 **Knowledge Base Management**
- Document statistics and analytics
- Real-time knowledge base status
- Document version history
- Chunk-level metadata tracking

### 🏠 **Local AI Processing**
- Uses Ollama for complete privacy (no cloud dependencies)
- Powered by Qwen2.5:7b language model
- nomic-embed-text for document embeddings
- No API keys required

## 🛠️ Technical Architecture

### Core Components

1. **AdvancedEncryptedVectorStore**: Main class handling document processing and storage
2. **Document Processing Pipeline**: PDF text extraction → chunking → embedding → encryption
3. **Smart Update Engine**: Version tracking and incremental updates
4. **Conversation Engine**: Query processing and response generation
5. **Streamlit UI**: User interface and interaction management

### Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**: Ollama (Qwen2.5:7b, nomic-embed-text)
- **Vector Database**: FAISS with encryption
- **Document Processing**: PyPDF2, LangChain
- **Security**: Cryptography (Fernet encryption)
- **Frontend**: Streamlit
- **Text Processing**: RecursiveCharacterTextSplitter

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.8 or higher**
2. **Ollama installed and running**

### 1. Install Ollama

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. Clone and Setup

```bash
# Navigate to project directory
cd Mem_Chat

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Upload Documents**
- Click "Upload your PDF files" in the sidebar
- Select one or multiple PDF files
- Click "🔄 Process Documents"

### 2. **Smart Updates**
- Re-upload modified documents - only changes will be processed
- The system automatically detects:
  - ✅ New documents
  - 🔄 Updated documents (only new content processed)
  - 📄 Unchanged documents (skipped)

### 3. **Chat with Documents**
- Type questions in the chat input
- Get AI-powered responses based on your document content
- View conversation history
- Clear chat history with "🗑 Clear Chat" button

### 4. **Monitor Knowledge Base**
- View document statistics in the sidebar
- See total documents and text chunks
- Check document list with last update timestamps

## 🔧 Configuration

### Model Configuration
The application uses these Ollama models by default:
- **Language Model**: `qwen2.5:7b` - For generating responses
- **Embedding Model**: `nomic-embed-text` - For document embeddings

### Storage Configuration
- **Storage Directory**: `encrypted_faiss_storage/` (created automatically)
- **Encryption**: AES-256 via Fernet (key auto-generated)
- **Chunk Size**: 500 characters with 200 character overlap

## 📁 Project Structure

```
Mem_Chat/
├── main.py                     # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── encrypted_faiss_storage/  # Encrypted vector store (auto-created)
    ├── key.key              # Encryption key
    ├── faiss_index.encrypted # Encrypted FAISS index
    ├── metadata.encrypted   # Encrypted chunk metadata
    └── doc_versions.encrypted # Document version tracking
```

## 🛡️ Security Features

- **Local Processing**: All AI processing happens locally
- **Encrypted Storage**: Document vectors and metadata encrypted at rest
- **No Cloud Dependencies**: No external API calls or data transmission
- **Secure Key Management**: Automatic encryption key generation and storage

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Models not found**
   ```bash
   # Pull required models
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```

3. **Port already in use**
   ```bash
   # Run on different port
   streamlit run main.py --server.port 8502
   ```

## 📝 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Multi-language support
- [ ] Advanced search and filtering
- [ ] Document comparison features
- [ ] Export conversation history
- [ ] Custom model selection
- [ ] Batch document processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Streamlit, LangChain, and Ollama**