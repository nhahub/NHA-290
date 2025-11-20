import streamlit as st
import torch
import sys
import traceback
import os

# Page config - must be first
st.set_page_config(
    page_title="Qwen2-0.5B-Instruct LoRA Chat",
    page_icon="🤖",
    layout="wide"
)

# Show loading state immediately
st.title("🤖 Qwen2-0.5B-Instruct LoRA Chat")

# Check imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import json
    st.success("✅ All packages imported successfully!")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("""
    Please install missing packages:
    ```bash
    pip install transformers peft torch accelerate sentencepiece
    ```
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
    background-color: #007bff;  # Blue
    color: white;
    border-radius: 12px;
}
.assistant-message {
    background-color: #e9ecef;  # Light gray
    color: black;
    border-radius: 12px;
}
    .message-content {
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer(checkpoint_path, use_checkpoint="checkpoint-7500"):
    """Load base model from HuggingFace and apply LoRA weights"""
    
    status_placeholder = st.empty()
    
    try:
        # First, try to load tokenizer from local checkpoint
        status_placeholder.info("🔄 Step 1/3: Loading tokenizer from local files...")
        
        adapter_path = os.path.join(checkpoint_path, use_checkpoint)
        
        if not os.path.exists(adapter_path):
            status_placeholder.error(f"❌ Checkpoint path not found: {adapter_path}")
            return None, None
        
        # Download tokenizer directly from HuggingFace
        status_placeholder.info("📥 Downloading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct",
            trust_remote_code=True
        )
        status_placeholder.success("✅ Loaded tokenizer from HuggingFace!")
        
        status_placeholder.info("🔄 Step 2/3: Downloading base model from HuggingFace...")
        
        # Load base model from HuggingFace
        base_model_name = "Qwen/Qwen2-0.5B-Instruct"
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        status_placeholder.info(f"🔄 Step 3/3: Loading LoRA adapter from {use_checkpoint}...")
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        
        status_placeholder.success("✅ Model loaded successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        status_placeholder.error(f"❌ Error loading model: {str(e)}")
        st.error("Full error traceback:")
        st.code(traceback.format_exc())
        
        # Show troubleshooting tips
        st.info("""
        **Troubleshooting:**
        
        1. Make sure you have internet connection (to download base model)
        2. Install required packages:
        ```bash
        pip install transformers peft torch accelerate sentencepiece protobuf
        ```
        3. Check that the model path is correct
        4. Ensure checkpoint folder exists
        """)
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate response from the model"""
    
    try:
        # Simple prompt format
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
        
        response = response.strip()
        
        return response
        
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        st.code(traceback.format_exc())
        return f"Error generating response: {str(e)}"

def main():
    st.markdown("Chat with your fine-tuned Qwen2 model")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model path input
        model_path = st.text_input(
            "Model Path",
            value=".",
            help="Path to your LoRA checkpoint folder (. = current directory)"
        )
        
        # Check if path exists
        if not os.path.exists(model_path):
            st.error(f"⚠️ Path not found: {model_path}")
            st.info("Current directory: " + os.getcwd())
        
        # Checkpoint selection
        available_checkpoints = []
        if os.path.exists(model_path):
            available_checkpoints = [d for d in os.listdir(model_path) 
                                    if d.startswith("checkpoint-")]
            available_checkpoints.sort()
        
        if available_checkpoints:
            checkpoint = st.selectbox(
                "Select Checkpoint",
                available_checkpoints,
                index=len(available_checkpoints)-1  # Select latest by default
            )
            st.success(f"✅ Found {len(available_checkpoints)} checkpoint(s)")
        else:
            checkpoint = "checkpoint-7500"
            st.warning("⚠️ No checkpoints found. Using default: checkpoint-7500")
        
        st.divider()
        
        # Generation parameters
        st.subheader("Generation Parameters")
        max_length = st.slider("Max Length", 50, 2048, 512, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"PyTorch: {torch.__version__}")
        st.text(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.text(f"GPU: {torch.cuda.get_device_name(0)}")
        
        st.divider()
        
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ Model path does not exist: {model_path}")
        st.info(f"Please update the path in the sidebar. Current working directory: {os.getcwd()}")
        return
    
    model, tokenizer = load_model_and_tokenizer(model_path, checkpoint)
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check the errors above.")
        return
    
    st.success("🎉 Model is ready! Start chatting below.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>👤 You:</b>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>🤖 Assistant:</b>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <b>👤 You:</b>
                <div class="message-content">{prompt}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(
                model, 
                tokenizer, 
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display assistant message
            with st.container():
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>🤖 Assistant:</b>
                    <div class="message-content">{response}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.code(traceback.format_exc())