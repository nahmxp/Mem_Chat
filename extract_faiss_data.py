import os
import json
import pickle
from datetime import datetime
from cryptography.fernet import Fernet
from langchain_community.llms import Ollama
import random
import re
from typing import List, Dict, Tuple

class FAISSDataExtractor:
    """
    Extracts training data from encrypted FAISS storage and generates
    question-answer pairs for fine-tuning a conversational model.
    """
    
    def __init__(self, storage_dir="encrypted_faiss_storage", output_dir="./training_data"):
        self.storage_dir = storage_dir
        self.output_dir = output_dir
        self.key_file = os.path.join(storage_dir, "key.key")
        self.metadata_file = os.path.join(storage_dir, "metadata.encrypted")
        self.document_versions_file = os.path.join(storage_dir, "doc_versions.encrypted")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load encryption cipher
        self.cipher = self._load_cipher()
        
        # Initialize QA generation model
        self.qa_generator = Ollama(model="qwen2.5:7b")
        
    def _load_cipher(self):
        """Load the encryption key."""
        if not os.path.exists(self.key_file):
            raise FileNotFoundError(f"Encryption key not found: {self.key_file}")
        
        with open(self.key_file, 'rb') as f:
            key = f.read()
        return Fernet(key)
    
    def _load_encrypted_data(self, file_path):
        """Load and decrypt data from an encrypted file."""
        if not os.path.exists(file_path):
            return {}
        
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return pickle.loads(decrypted_data)
        except Exception as e:
            print(f"Error loading encrypted data from {file_path}: {e}")
            return {}
    
    def extract_document_chunks(self) -> List[Dict]:
        """
        Extract all document chunks from encrypted FAISS storage.
        Returns a list of chunk dictionaries with text and metadata.
        """
        print("Extracting document chunks from FAISS storage...")
        
        # Load chunk metadata
        chunk_metadata = self._load_encrypted_data(self.metadata_file)
        
        # Load document versions for additional context
        document_versions = self._load_encrypted_data(self.document_versions_file)
        
        chunks = []
        for chunk_id, chunk_data in chunk_metadata.items():
            chunk_info = {
                'id': chunk_id,
                'text': chunk_data.get('text', ''),
                'doc_name': chunk_data.get('doc_name', ''),
                'chunk_index': chunk_data.get('chunk_index', 0),
                'timestamp': chunk_data.get('timestamp', ''),
                'hash': chunk_data.get('hash', '')
            }
            chunks.append(chunk_info)
        
        print(f"Extracted {len(chunks)} chunks from {len(document_versions)} documents")
        return chunks
    
    def generate_questions_from_chunk(self, chunk_text: str, doc_name: str) -> List[str]:
        """
        Generate multiple questions that could be answered by the given chunk.
        """
        prompt = f"""Based on the following text from document "{doc_name}", generate 3-5 diverse questions that this text could answer. 

Text: {chunk_text}

Generate questions that are:
1. Specific and answerable from the text
2. Varied in style (factual, analytical, explanatory)
3. Natural and conversational
4. Different levels of complexity

Format as a numbered list:
1. [Question 1]
2. [Question 2]
3. [Question 3]
etc.
"""
        
        try:
            response = self.qa_generator.invoke(prompt)
            
            # Extract questions from the response
            questions = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Match numbered list items
                if re.match(r'^\d+\.', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:5]  # Limit to 5 questions max
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def improve_answer_with_context(self, question: str, chunk_text: str, doc_name: str) -> str:
        """
        Generate an improved, conversational answer based on the chunk text.
        """
        prompt = f"""You are an intelligent assistant. Answer the following question based on the provided context from document "{doc_name}".

Question: {question}

Context: {chunk_text}

Provide a clear, concise, and helpful answer. Be conversational but accurate. If the context doesn't fully answer the question, say so and provide what information you can.

Answer:"""
        
        try:
            response = self.qa_generator.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating improved answer: {e}")
            return chunk_text  # Fallback to original chunk
    
    def generate_qa_pairs(self, chunks: List[Dict], max_pairs_per_chunk: int = 3) -> List[Dict]:
        """
        Generate question-answer pairs from document chunks.
        """
        print("Generating question-answer pairs...")
        qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_text = chunk['text']
            doc_name = chunk['doc_name']
            
            # Skip very short chunks
            if len(chunk_text.split()) < 10:
                continue
            
            # Generate questions for this chunk
            questions = self.generate_questions_from_chunk(chunk_text, doc_name)
            
            # Create QA pairs
            for question in questions[:max_pairs_per_chunk]:
                # Generate improved answer
                answer = self.improve_answer_with_context(question, chunk_text, doc_name)
                
                qa_pair = {
                    'question': question,
                    'answer': answer,
                    'source_doc': doc_name,
                    'source_chunk_id': chunk['id'],
                    'original_text': chunk_text,
                    'timestamp': datetime.now().isoformat()
                }
                qa_pairs.append(qa_pair)
        
        print(f"Generated {len(qa_pairs)} question-answer pairs")
        return qa_pairs
    
    def augment_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Augment QA pairs with variations and additional examples.
        """
        print("Augmenting QA pairs with variations...")
        augmented_pairs = qa_pairs.copy()
        
        for qa_pair in qa_pairs[:100]:  # Limit augmentation to avoid too much data
            original_question = qa_pair['question']
            original_answer = qa_pair['answer']
            
            # Generate question variations
            variation_prompt = f"""Rephrase this question in 2 different ways while keeping the same meaning:

Original: {original_question}

Provide 2 alternative phrasings:
1. [Variation 1]
2. [Variation 2]
"""
            
            try:
                response = self.qa_generator.invoke(variation_prompt)
                lines = response.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        variation = re.sub(r'^\d+\.\s*', '', line).strip()
                        if variation and variation != original_question:
                            augmented_pair = qa_pair.copy()
                            augmented_pair['question'] = variation
                            augmented_pair['is_augmented'] = True
                            augmented_pairs.append(augmented_pair)
                            
            except Exception as e:
                print(f"Error in augmentation: {e}")
                continue
        
        print(f"Augmented dataset size: {len(augmented_pairs)}")
        return augmented_pairs
    
    def save_training_data(self, qa_pairs: List[Dict], filename: str = "faiss_training_data.json"):
        """
        Save QA pairs in the format expected by the training script.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Format for training
        training_data = []
        for qa_pair in qa_pairs:
            training_example = {
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'metadata': {
                    'source_doc': qa_pair.get('source_doc', ''),
                    'source_chunk_id': qa_pair.get('source_chunk_id', ''),
                    'timestamp': qa_pair.get('timestamp', ''),
                    'is_augmented': qa_pair.get('is_augmented', False)
                }
            }
            training_data.append(training_example)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to: {output_path}")
        print(f"Total training examples: {len(training_data)}")
        
        # Save statistics
        stats = {
            'total_examples': len(training_data),
            'source_documents': len(set(qa['metadata']['source_doc'] for qa in training_data)),
            'augmented_examples': len([qa for qa in training_data if qa['metadata'].get('is_augmented', False)]),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        return output_path
    
    def create_training_dataset(self, max_pairs_per_chunk: int = 3, augment: bool = True):
        """
        Complete pipeline to create training dataset from FAISS storage.
        """
        print("=" * 60)
        print("CREATING TRAINING DATASET FROM FAISS STORAGE")
        print("=" * 60)
        
        # Extract chunks
        chunks = self.extract_document_chunks()
        
        if not chunks:
            print("No chunks found in FAISS storage. Please upload some documents first.")
            return None
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(chunks, max_pairs_per_chunk)
        
        if not qa_pairs:
            print("No QA pairs generated. Check your documents and try again.")
            return None
        
        # Augment if requested
        if augment:
            qa_pairs = self.augment_qa_pairs(qa_pairs)
        
        # Save training data
        output_path = self.save_training_data(qa_pairs)
        
        print("=" * 60)
        print("TRAINING DATASET CREATION COMPLETED!")
        print(f"Dataset saved to: {output_path}")
        print("=" * 60)
        
        return output_path

if __name__ == "__main__":
    # Create extractor
    extractor = FAISSDataExtractor()
    
    # Create training dataset
    dataset_path = extractor.create_training_dataset(
        max_pairs_per_chunk=3,
        augment=True
    )
    
    if dataset_path:
        print(f"Training dataset ready at: {dataset_path}")
        print("You can now use this with train_conversational.py")