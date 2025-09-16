# Fine-Tuning Workflow Guide 🎯

This guide explains how to create a self-contained AI model that has learned directly from your document content, eliminating the need for vector retrieval.

## 🎯 Overview

Instead of using RAG (Retrieval-Augmented Generation) which searches for relevant document chunks, fine-tuning creates a model that has "memorized" your document content and can answer questions directly.

### RAG vs Fine-tuning Comparison

| Aspect | RAG Approach | Fine-tuned Approach |
|--------|--------------|---------------------|
| **Speed** | Slower (search + generate) | Faster (direct generation) |
| **Memory** | Low model memory, high storage | High model memory, low storage |
| **Updates** | Easy (just add documents) | Requires retraining |
| **Accuracy** | Good with citations | Excellent for learned content |
| **Resource Usage** | CPU/GPU + Vector DB | GPU for inference only |

## 📋 Step-by-Step Workflow

### Step 1: Upload Documents to Main App
First, make sure you have documents in your FAISS storage:

```bash
# Run the main app and upload PDFs
python main.py
```

Upload your PDF documents and process them to build the knowledge base.

### Step 2: Extract Training Data from FAISS
Convert your encrypted FAISS storage into training data:

```bash
python extract_faiss_data.py
```

This will:
- ✅ Read your encrypted document chunks
- ✅ Generate 3-5 questions per chunk using Qwen2.5
- ✅ Create improved answers for each question
- ✅ Augment data with question variations
- ✅ Save training data to `./training_data/faiss_training_data.json`

### Step 3: Fine-tune the Model
Train a conversational model on your document content:

```bash
python train_conversational.py
```

**Training Configuration:**
- **Model**: Microsoft DialoGPT-small (optimized for conversation)
- **Epochs**: 15 (for thorough learning)
- **Batch Size**: 4 per device + 8 gradient accumulation = 32 effective
- **Learning Rate**: 3e-4 (higher for faster convergence)
- **Memory**: Optimized for RTX 4060 Ti 16GB

**Expected Training Time**: 2-4 hours depending on document size

### Step 4: Test the Fine-tuned Model
Once training is complete, test your model:

```bash
# Command-line interface
python test_finetuned_model.py

# Or Streamlit interface
python -m streamlit run test_finetuned_model.py streamlit
```

### Step 5: Use Hybrid App (Optional)
Use the hybrid app that supports both RAG and fine-tuned approaches:

```bash
python -m streamlit run main_hybrid.py
```

## 📁 File Structure After Fine-tuning

```
Mem_Chat/
├── main.py                           # Original RAG-based app
├── main_hybrid.py                    # Hybrid app (RAG + Fine-tuned)
├── extract_faiss_data.py            # Data extraction from FAISS
├── train_conversational.py          # Fine-tuning script
├── test_finetuned_model.py          # Model testing and inference
├── encrypted_faiss_storage/         # Original knowledge base
│   ├── key.key
│   ├── faiss_index.encrypted
│   ├── metadata.encrypted
│   └── doc_versions.encrypted
├── training_data/                   # Generated training data
│   ├── faiss_training_data.json
│   └── training_stats.json
└── finetuned_model/                # Your trained model
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── ...
```

## ⚙️ Configuration Options

### Extract Data Configuration

```python
# In extract_faiss_data.py
extractor = FAISSDataExtractor()
dataset_path = extractor.create_training_dataset(
    max_pairs_per_chunk=3,    # Questions per chunk
    augment=True              # Create question variations
)
```

### Training Configuration

```python
# In train_conversational.py
trainer_instance = ConversationalTrainer(
    model_name="microsoft/DialoGPT-small",           # Base model
    dataset_path="./training_data/faiss_training_data.json"
)
```

### Inference Configuration

```python
# In test_finetuned_model.py
response = model.generate_response(
    question,
    max_length=512,      # Response length
    temperature=0.7,     # Creativity (0.1-1.0)
    do_sample=True       # Use sampling
)
```

## 🚀 Usage Scenarios

### Scenario 1: Fast Responses
Use fine-tuned model when you need:
- Quick responses (no vector search overhead)
- Offline operation (no need for vector database)
- Consistent performance regardless of query complexity

### Scenario 2: Up-to-date Information
Use RAG when you need:
- Frequently updated documents
- Ability to add new documents without retraining
- Source citations and transparency

### Scenario 3: Hybrid Approach
Use both models:
- Fine-tuned for general knowledge about your documents
- RAG for specific facts and recent updates

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory During Training**
   ```python
   # Reduce batch size in train_conversational.py
   per_device_train_batch_size=2  # Instead of 4
   gradient_accumulation_steps=4  # Instead of 8
   ```

2. **No Training Data Generated**
   - Ensure documents are uploaded to main.py first
   - Check if FAISS storage exists in `encrypted_faiss_storage/`

3. **Model Not Loading**
   - Verify training completed successfully
   - Check if `finetuned_model/` directory exists

4. **Poor Model Performance**
   - Increase training epochs (15 → 25)
   - Generate more training data (increase `max_pairs_per_chunk`)
   - Use a larger base model (DialoGPT-medium instead of small)

### Performance Optimization

**For Better Accuracy:**
- Use more documents for training
- Increase `max_pairs_per_chunk` to 5
- Enable augmentation for more diverse examples

**For Faster Training:**
- Use a smaller model (DialoGPT-small)
- Reduce max_length to 256
- Disable gradient checkpointing

**For Memory Efficiency:**
- Enable gradient checkpointing
- Use gradient accumulation
- Reduce batch size

## 📊 Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./finetuned_model/runs
```

Key metrics to watch:
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Should follow cosine schedule

## 🎯 Next Steps

1. **Evaluate Performance**: Compare RAG vs fine-tuned responses
2. **Optimize Hyperparameters**: Adjust based on your specific use case
3. **Scale Up**: Use larger models for better performance
4. **Deploy**: Create production-ready inference API

---

**🎉 Congratulations!** You now have a self-contained AI model trained on your specific documents!