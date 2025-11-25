"""
SQA Project Streamlit Web Interface
Modern chatbot interface
"""
import sys
import os
from pathlib import Path
import json
import streamlit as st
from PIL import Image
import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.data.data_loader import ScienceQADataLoader
from src.multimodal.llava_processor import LLaVAImageProcessor
from src.llm.qwen_model import QwenLLM
from src.rag.vector_store import ScienceQAVectorStore
from src.rag.rag_system import ScienceQARAGSystem

# Page configuration
st.set_page_config(
    page_title="ScienceQA RAG Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styles - modern dark theme
st.markdown("""
<style>
    /* 主容器样式 */
    .main-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 1rem;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 0.25rem;
    }
    
    .chat-message.assistant {
        background: rgba(255, 255, 255, 0.1);
        color: #1080E7;
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
        backdrop-filter: blur(10px);
    }
    
    .chat-message.assistant strong {
        color: #1080E7;
    }
    
    /* 系统状态卡片 */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .status-item:last-child {
        border-bottom: none;
    }
    
    .status-label {
        color: #b0b0b0;
        font-size: 0.9rem;
    }
    
    .status-value {
        color: #4ade80;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* 标题样式 */
    .chat-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chat-subtitle {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-bottom: 1.5rem;
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #888;
    }
    
    /* 文件上传器样式 - 嵌入到输入框内，紧凑设计 */
    .stFileUploader {
        width: 100% !important;
        margin: 0 !important;
    }
    
    .stFileUploader>div {
        width: 100% !important;
        margin: 0 !important;
    }
    
    .stFileUploader>div>div {
        width: 100% !important;
        margin: 0 !important;
    }
    
    .stFileUploader>div>div>div {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 2px dashed rgba(255, 255, 255, 0.25) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        min-height: 80px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stFileUploader>div>div>div:hover {
        background: rgba(255, 255, 255, 0.18) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    .stFileUploader label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin: 0 0 0.25rem 0 !important;
        padding: 0 !important;
        display: block !important;
    }
    
    .stFileUploader small {
        color: rgba(255, 255, 255, 0.65) !important;
        font-size: 0.75rem !important;
        margin: 0.25rem 0 0.5rem 0 !important;
        display: block !important;
    }
    
    .stFileUploader button {
        background: rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 0.4rem !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.85rem !important;
        margin-top: 0.25rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .stFileUploader button:hover {
        background: rgba(255, 255, 255, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-1px);
    }
    
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 控制台样式 */
    .console-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 活动指示器 */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
        background-color: #4ade80;
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.6);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)"""
    try:
        # Load vector database
        vector_store = ScienceQAVectorStore()
        vector_store.load_vector_store()
        
        # Initialize LLM
        llm = QwenLLM()
        
        # Load problems and captions data
        loader = ScienceQADataLoader()
        problems = loader.load_problems()
        captions = loader.load_captions()
        
        # Load LLaVA-generated descriptions (if exists)
        llava_captions = None
        llava_captions_path = config.DATA_DIR / "llava_captions.json"
        if llava_captions_path.exists():
            with open(llava_captions_path, 'r', encoding='utf-8') as f:
                llava_captions = json.load(f)
        
        # Create RAG system
        rag_system = ScienceQARAGSystem(
            vector_store, 
            llm,
            problems=problems,
            captions=captions,
            llava_captions=llava_captions
        )
        
        return rag_system, None
    except FileNotFoundError as e:
        return None, f"Vector database not found: {e}\nPlease run 'python main.py build_db' to build vector database first"
    except Exception as e:
        return None, f"Failed to load RAG system: {str(e)}"


def init_session_state():
    """Initialize session state"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'rag_error' not in st.session_state:
        st.session_state.rag_error = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'llava_processor' not in st.session_state:
        st.session_state.llava_processor = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


def main():
    """Main function"""
    init_session_state()
    
    # Load RAG system
    if st.session_state.rag_system is None:
        try:
            with st.spinner("Loading RAG system..."):
                rag_system, error = load_rag_system()
                if error:
                    st.session_state.rag_error = error
                    st.error(error)
                    st.info("💡 Tip: Please run `python main.py build_db` to build vector database first")
                    st.session_state.rag_system = None
                else:
                    st.session_state.rag_system = rag_system
        except Exception as e:
            st.error(f"Error loading RAG system: {str(e)}")
            import traceback
            with st.expander("View detailed error"):
                st.code(traceback.format_exc())
            st.session_state.rag_system = None
    
    # Check if RAG system is available
    if st.session_state.rag_system is None:
        st.warning("⚠️ RAG system not loaded, cannot perform Q&A. Please build vector database first.")
        if st.button("🔄 Retry loading"):
            st.session_state.rag_system = None
            st.session_state.rag_error = None
            st.rerun()
        return
    
    rag_system = st.session_state.rag_system
    
    # Initialize welcome message
    if not st.session_state.initialized:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Hello! I am the ScienceQA intelligent assistant. I can help you answer science questions, supporting multimodal Q&A with text and images. Please feel free to ask your questions!",
            "timestamp": datetime.datetime.now().strftime("%H:%M")
        }]
        st.session_state.initialized = True
    
    # Create two-column layout: left chat, right console
    col_chat, col_console = st.columns([2, 1])
    
    with col_chat:
        # Chat interface title
        st.markdown("""
        <div class="chat-title">
            <span>🔬 ScienceQA RAG Chatbot</span>
            <span class="status-indicator"></span>
        </div>
        <div class="chat-subtitle">Based on LLaVA + LangChain + RAG</div>
        """, unsafe_allow_html=True)
        
        # Chat history area
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg.get("role") == "assistant":
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <strong style="color: #1080E7;">ScienceQA Assistant</strong><br>
                        <span style="color: #1080E7;">{msg.get("content", "")}</span>
                        <div style="font-size: 0.75rem; color: #888; margin-top: 0.5rem;">{msg.get("timestamp", "")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif msg.get("role") == "user":
                    question_text = msg.get("content", "")
                    if msg.get("is_image_question"):
                        question_text = "🖼️ " + question_text
                    st.markdown(f"""
                    <div class="chat-message user">
                        <strong>You</strong><br>
                        {question_text}
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); margin-top: 0.5rem;">{msg.get("timestamp", "")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    # If there's an answer, display it
                    if msg.get("answer"):
                        st.markdown(f"""
                        <div class="chat-message assistant" style="margin-top: 0.5rem;">
                            <strong style="color: #1080E7;">ScienceQA Assistant</strong><br>
                            <span style="color: #1080E7;">{msg.get("answer", "")}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Input area - file upload component embedded in input box
        # Create unified input container: file upload (left) + input box (middle) + send button (right)
        input_col1, input_col2, input_col3 = st.columns([2.5, 6.5, 1])
        
        with input_col1:
            uploaded_image = st.file_uploader(
                "Drag and drop file here",
                type=['png', 'jpg', 'jpeg'],
                help="Upload image",
                key="image_upload"
            )
        
        with input_col2:
            question = st.text_input(
                "Please enter your question...",
                key="question_input",
                label_visibility="collapsed",
                placeholder="Please enter your question..."
            )
        
        with input_col3:
            send_button = st.button("Send", type="primary", use_container_width=True, key="send_btn")
        
        st.caption("Enter to send · Shift + Enter for new line | Supports text and image multimodal Q&A")
    
    with col_console:
        # System console
        st.markdown("""
        <div class="console-header">
            <span>⚙️ System Console</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": "Hello! I am the ScienceQA intelligent assistant. I can help you answer science questions, supporting multimodal Q&A with text and images. Please feel free to ask your questions!",
                "timestamp": datetime.datetime.now().strftime("%H:%M")
            }]
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # System status
        st.markdown("""
        <div class="status-card">
            <div class="status-item">
                <span class="status-label">RAG Knowledge Base</span>
                <span class="status-value">✅ Connected</span>
            </div>
            <div class="status-item">
                <span class="status-label">LLaVA Model</span>
                <span class="status-value">✅ Running</span>
            </div>
            <div class="status-item">
                <span class="status-label">LangChain</span>
                <span class="status-value">✅ Ready</span>
            </div>
            <div class="status-item">
                <span class="status-label">Messages</span>
                <span class="status-value">📄 {count}</span>
            </div>
        </div>
        """.format(count=len(st.session_state.chat_history)), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model configuration (collapsible)
        with st.expander("⚙️ Model Configuration"):
            top_k = st.slider("Retrieval Documents", 1, 10, config.TOP_K_RETRIEVAL, key="top_k_slider")
            temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.1, key="temp_slider")
            subject_filter = st.selectbox(
                "Subject Filter",
                ["All", "biology", "physics", "chemistry", "earth science", "geography", "history"],
                key="subject_filter"
            )
        
        # Knowledge base statistics (collapsible)
        with st.expander("📊 Knowledge Base Statistics"):
            st.text(f"Vector Database: ✅")
            st.text(f"GPU Count: {config.NUM_GPUS}")
            st.text(f"Retrieval Documents: {top_k}")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            <div>版本 v1.0.0</div>
            <div style="margin-top: 0.5rem;">Powered by AI Technology</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process user input
    if send_button and question:
        current_time = datetime.datetime.now().strftime("%H:%M")
        
        # Add user message
        user_msg = {
            "role": "user",
            "content": question,
            "timestamp": current_time,
            "is_image_question": False
        }
        
        # If image is uploaded
        image_path = None
        if uploaded_image:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_image.read())
                image_path = tmp_file.name
            user_msg["is_image_question"] = True
        
        st.session_state.chat_history.append(user_msg)
        
        # Process question
        with st.spinner("Thinking..."):
            try:
                subject = None if subject_filter == "All" else subject_filter
                
                if not question:
                    question = "Please analyze this image and answer the question."
                
                # Retrieve and generate answer
                result = rag_system.answer_with_rag(
                    question=question,
                    k=top_k,
                    subject=subject,
                    image_path=image_path
                )
                
                # Add assistant reply
                answer_text = result["answer"]
                if result.get("is_image_question"):
                    answer_text = "🖼️ Image question detected, multimodal model processing completed\n\n" + answer_text
                
                assistant_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "timestamp": current_time,
                    "answer": result["answer"],
                    "retrieved_docs": result.get("retrieved_documents", 0),
                    "is_image_question": result.get("is_image_question", False)
                }
                
                st.session_state.chat_history.append(assistant_msg)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                import traceback
                with st.expander("View detailed error"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

