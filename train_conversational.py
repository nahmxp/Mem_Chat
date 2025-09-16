"""
Fine-tuning script for creating a self-contained conversational AI model
from FAISS vector store data.

This script trains a language model on question-answer pairs extracted
from your encrypted FAISS storage, creating a model that can answer
questions about your documents without needing vector retrieval.

Usage:
1. First run: python extract_faiss_data.py (to generate training data)
2. Then run: python train_conversational.py (to fine-tune the model)

The resulting model will be saved to ./finetuned_model/
"""

# Standard library imports
import json
import os
import time

# Third-party imports
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

class ConversationalTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-small", dataset_path="./training_data/faiss_training_data.json"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = "./finetuned_model"
        
        # Check CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def load_model_and_tokenizer(self):
        print(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for conversation
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
        }
        
        # Add tokens if they don't exist
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added_tokens} special tokens")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,  # Use float32 to avoid FP16 issues
            device_map="auto",
        )
        
        # Resize token embeddings if needed
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Model loaded successfully")
        return self.model, self.tokenizer
    
    def prepare_dataset(self):
        print(f"Loading dataset from: {self.dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found at {self.dataset_path}")
            print("Please run extract_faiss_data.py first to generate training data from your FAISS storage.")
            raise FileNotFoundError(f"Training dataset not found: {self.dataset_path}")
        
        # Load the JSON dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples from JSON")
        
        # Check data structure
        if data and isinstance(data[0], dict):
            if 'question' in data[0] and 'answer' in data[0]:
                # Data is already in question-answer format
                conversation_texts = []
                for item in data:
                    question = item.get('question', '').strip()
                    answer = item.get('answer', '').strip()
                    
                    if question and answer:
                        # Format as conversation with document context
                        source_doc = item.get('metadata', {}).get('source_doc', 'Unknown Document')
                        conversation_text = f"Human: {question}\nAssistant: Based on the document '{source_doc}': {answer}"
                        conversation_texts.append(conversation_text)
                
                print(f"Formatted {len(conversation_texts)} conversation examples")
            else:
                raise ValueError("Dataset format not recognized. Expected 'question' and 'answer' fields.")
        else:
            raise ValueError("Invalid dataset structure")
        
        # Create dataset
        dataset = Dataset.from_dict({"text": conversation_texts})
        
        def tokenize_function(examples):
            # Add conversation formatting with special tokens
            formatted_texts = []
            for text in examples['text']:
                # Add special tokens for conversation structure
                formatted_text = f"<|startoftext|>{text}<|endoftext|>"
                formatted_texts.append(formatted_text)
            
            # Tokenize
            tokenized = self.tokenizer(
                formatted_texts,
                truncation=True,
                padding=True,
                max_length=512,  # Adjust based on your data
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Split into train/validation (90/10 split)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Training examples: {len(train_test_split['train'])}")
        print(f"Validation examples: {len(train_test_split['test'])}")
        
        return train_test_split['train'], train_test_split['test']
    
    def setup_training_arguments(self):
        """Optimized training arguments for RTX 4060 Ti 16GB"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            
            # Training hyperparameters - optimized for high learning and many epochs
            num_train_epochs=15,              # Many epochs as requested
            learning_rate=3e-4,               # Higher learning rate for faster convergence
            warmup_steps=100,                 # Warm up learning rate
            weight_decay=0.01,                # Regularization
            
            # Batch sizes optimized for 16GB VRAM
            per_device_train_batch_size=4,    # Adjust based on memory usage
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,    # Effective batch size = 4 * 8 = 32
            
            # Memory optimization
            fp16=False,                       # Disable FP16 to avoid gradient scaling issues
            bf16=True if torch.cuda.is_bf16_supported() else False,  # Use BF16 if available
            dataloader_pin_memory=True,
            gradient_checkpointing=True,      # Trade compute for memory
            
            # Evaluation and logging
            eval_strategy="steps",            # Updated parameter name
            eval_steps=50,                    # Evaluate every 50 steps
            logging_steps=25,                 # Log every 25 steps
            save_strategy="steps",
            save_steps=100,                   # Save checkpoint every 100 steps
            
            # Early stopping and best model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Optimization
            optim="adamw_torch",              # AdamW optimizer
            lr_scheduler_type="cosine",       # Cosine learning rate schedule
            
            # Misc
            report_to="tensorboard",          # Tensorboard logging
            run_name=f"conversational_finetune_{int(time.time())}",
            seed=42,
            
            # Performance
            dataloader_num_workers=2,         # Parallel data loading
            remove_unused_columns=False,
        )
        
        return training_args
    
    def train(self):
        print("=" * 60)
        print("STARTING CONVERSATIONAL AI FINE-TUNING")
        print("=" * 60)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficiency on modern GPUs
        )
        
        # Setup trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Starting training...")
        print(f"Total training examples: {len(train_dataset)}")
        print(f"Total validation examples: {len(eval_dataset)}")
        print(f"Training for {training_args.num_train_epochs} epochs")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        
        # Train the model
        trainer.train()
        
        # Save the final model
        print("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        # Save training state
        trainer.save_state()
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to: {self.output_dir}")
        print("=" * 60)
        
        return trainer

if __name__ == "__main__":
    # Create trainer instance
    trainer_instance = ConversationalTrainer()
    
    # Start training
    trainer = trainer_instance.train()
