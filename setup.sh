#!/bin/bash
# Setup script for Mem_Chat with CUDA support

echo "=== Mem_Chat Setup Script ==="
echo "Setting up Python environment with CUDA support..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Linux/Mac
    source .venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other dependencies..."
pip install streamlit>=1.28.0 langchain>=0.1.0 langchain-community>=0.0.20 langchain-ollama>=0.1.0 faiss-cpu>=1.7.4 PyPDF2>=3.0.1 cryptography>=41.0.0 ollama>=0.1.7 transformers>=4.30.0 datasets>=2.14.0 accelerate>=0.20.0 tensorboard>=2.13.0

# Verify CUDA installation
echo "Verifying CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

echo "=== Setup Complete! ==="
echo "Run 'python main.py' to start the application"