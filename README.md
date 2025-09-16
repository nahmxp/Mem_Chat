# Mem_Chat - Advanced PDF Chatbot with Smart Updates 🤖📖

An intelligent PDF document chatbot that features smart incremental updates, encrypted storage, and local AI processing using Ollama.

## 🎯 Project Overview

Mem_Chat is an advanced PDF chatbot application that allows users to upload PDF documents and have intelligent conversations with their content. The key innovation is its **smart update system** that only processes new or changed content when documents are re-uploaded, making it highly efficient for large document collections.

## ✨ Key Features

### 🔒 **Encrypted Vector Storage**
- Dual backend support: FAISS (performance) or SQLite+Vec (mobile-friendly)
- All document embeddings and metadata are encrypted at rest
- Secure key management with automatic key generation

### 🧠 **Smart Document Updates**
- **Incremental Processing**: Only processes new or changed content
- **Version Tracking**: Maintains document versions and change history
- **Hash-based Detection**: Uses SHA256 hashing to detect document changes
- **Chunk-level Updates**: Identifies and updates only modified text chunks

### 🎯 **Dual AI Approaches**
- **RAG (Retrieval-Augmented Generation)**: Traditional vector search + generation
- **Fine-tuned Model**: Self-contained model trained on your documents
- **Hybrid Mode**: Choose between approaches or use both

### 🗄️ **Flexible Storage Backends**
- **FAISS**: Maximum performance for large datasets (desktop/server)
- **SQLite+Vec**: Mobile-friendly with excellent Flutter integration
- **Encrypted Storage**: All data encrypted at rest with both backends

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

### Storage Backend Options

**FAISS Backend** (`main.py`)
- **Best for**: Desktop applications requiring maximum performance
- **Advantages**: Ultra-fast vector similarity search, optimal for large document sets
- **Use when**: Performance is critical, desktop-only deployment

**SQLite+Vec Backend** (`main_sqlite.py`)  
- **Best for**: Mobile apps, cross-platform deployment, Flutter integration
- **Advantages**: Self-contained database, mobile-friendly, SQL compatibility
- **Use when**: Flutter/mobile integration needed, simpler deployment required

**Hybrid Backend** (`main_hybrid.py`)
- **Best for**: Applications needing both RAG and fine-tuned model responses
- **Advantages**: Combines retrieval-augmented generation with custom fine-tuned models
- **Use when**: Maximum answer quality and variety needed

### Backend Comparison

| Feature | FAISS | SQLite+Vec |
|---------|-------|------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mobile Support | ❌ | ✅ |
| Setup Complexity | Medium | Low |
| Dependencies | NumPy, FAISS | SQLite only |
| File Size | Larger | Smaller |
| Query Speed | Fastest | Fast |
| Flutter Integration | ❌ | ✅ |

### Core Components

1. **AdvancedEncryptedVectorStore**: Main class handling document processing and storage
2. **Document Processing Pipeline**: PDF text extraction → chunking → embedding → encryption
3. **Smart Update Engine**: Version tracking and incremental updates
4. **Conversation Engine**: Query processing and response generation
5. **Streamlit UI**: User interface and interaction management

### Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**: Ollama (Qwen2.5:7b, nomic-embed-text)
- **Vector Database**: FAISS with encryption / SQLite+Vec for mobile
- **Document Processing**: PyPDF2, LangChain, python-docx
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

# Easy setup with provided scripts:
# Windows:
setup.bat

# Linux/Mac:
chmod +x setup.sh
./setup.sh

# Or manual installation:
pip install -r requirements.txt
```

### 3. Choose Your Backend and Run

#### **FAISS Version** (Maximum Performance)
```bash
streamlit run main.py
```

#### **SQLite+Vec Version** (Flutter-Friendly)
```bash
streamlit run main_sqlite.py
```

#### **Hybrid Version** (Both RAG + Fine-tuned)
```bash
streamlit run main_hybrid.py
```

The application will open in your browser (usually `http://localhost:8501` or `8502`)

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

## 🎯 Fine-Tuning Your Own Model

For even better performance, you can create a self-contained model trained specifically on your documents:

### Quick Start Fine-Tuning

```bash
# 1. Extract training data from your knowledge base
python extract_faiss_data.py

# 2. Fine-tune a model on your documents
python train_conversational.py

# 3. Test your fine-tuned model
python test_finetuned_model.py

# 4. Use hybrid app with both approaches
streamlit run main_hybrid.py
```

**Benefits of Fine-Tuning:**
- ⚡ **Faster responses** (no vector search needed)
- 🧠 **Better understanding** of your specific domain
- 🔄 **Self-contained** model that works offline
- 📈 **Improved accuracy** for domain-specific questions

For detailed fine-tuning instructions, see [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)

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
├── main.py                     # FAISS-based application (desktop)
├── main_sqlite.py             # SQLite+Vec version (mobile-friendly)
├── main_hybrid.py             # Hybrid app (RAG + Fine-tuned)
├── extract_faiss_data.py      # Extract training data from FAISS
├── train_conversational.py    # Fine-tuning script
├── test_finetuned_model.py   # Fine-tuned model inference
├── setup.bat                  # Windows setup script
├── setup.sh                   # Linux/Mac setup script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── FINETUNING_GUIDE.md      # Detailed fine-tuning guide
├── encrypted_faiss_storage/  # FAISS encrypted vector store (auto-created)
│   ├── key.key              # Encryption key
│   ├── faiss_index.encrypted # Encrypted FAISS index
│   ├── metadata.encrypted   # Encrypted chunk metadata
│   └── doc_versions.encrypted # Document version tracking
├── encrypted_sqlite_storage/ # SQLite+Vec encrypted store (auto-created)
│   ├── key.key              # Encryption key
│   ├── vector_db.encrypted  # Encrypted SQLite database
│   └── doc_versions.encrypted # Document version tracking
├── training_data/           # Generated training data (auto-created)
│   ├── faiss_training_data.json
│   └── training_stats.json
└── finetuned_model/        # Your trained model (after fine-tuning)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
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