"""
Inference script for the fine-tuned conversational model.
This allows you to chat with your fine-tuned model that has learned
from your document knowledge base.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

class FineTunedModelInference:
    def __init__(self, model_path="./finetuned_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print(f"Loading fine-tuned model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_response(self, question: str, max_length: int = 512, temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        Generate a response to a question using the fine-tuned model.
        """
        if not self.model or not self.tokenizer:
            return "Model not loaded properly."
        
        try:
            # Format the input as a conversation
            input_text = f"<|startoftext|>Human: {question}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            else:
                response = full_response.replace(input_text, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def chat_interface(self):
        """Simple command-line chat interface."""
        print("=" * 60)
        print("FINE-TUNED MODEL CHAT INTERFACE")
        print("=" * 60)
        print("Type 'quit' to exit")
        print()
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("Assistant: ", end="")
            response = self.generate_response(question)
            print(response)
            print()

def create_streamlit_app():
    """
    Create a Streamlit app for the fine-tuned model.
    This can be used as an alternative to the RAG-based approach.
    """
    st.set_page_config(page_title="Fine-Tuned Document AI", page_icon="🧠")
    
    st.title("🧠 Fine-Tuned Document AI")
    st.markdown("Chat with your fine-tuned model that has learned from your documents!")
    
    # Initialize model in session state
    if 'finetuned_model' not in st.session_state:
        with st.spinner("Loading fine-tuned model..."):
            try:
                st.session_state.finetuned_model = FineTunedModelInference()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
    
    # Chat interface
    if "ft_messages" not in st.session_state:
        st.session_state.ft_messages = [
            {"role": "assistant", "content": "Hello! I'm your fine-tuned AI that has learned from your documents. Ask me anything!"}
        ]
    
    # Display chat history
    for message in st.session_state.ft_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.ft_messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.finetuned_model.generate_response(prompt)
            st.write(response)
        
        # Add assistant response
        st.session_state.ft_messages.append({"role": "assistant", "content": response})
    
    # Sidebar with model info
    with st.sidebar:
        st.header("🧠 Model Info")
        st.info("This is your fine-tuned model that has learned directly from your document content.")
        
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        max_length = st.slider("Max Response Length", 128, 1024, 512, 64)
        
        # Update model settings
        if 'finetuned_model' in st.session_state:
            # Store settings for next generation
            st.session_state.model_temperature = temperature
            st.session_state.model_max_length = max_length
        
        if st.button("🗑 Clear Chat"):
            st.session_state.ft_messages = [
                {"role": "assistant", "content": "Hello! I'm your fine-tuned AI that has learned from your documents. Ask me anything!"}
            ]
            st.rerun()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Run command-line interface
        model = FineTunedModelInference()
        model.chat_interface()