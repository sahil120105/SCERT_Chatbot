# app.py
import streamlit as st
from src.rag_engine import MarathiRAG

# Page Config
st.set_page_config(page_title="SCERT Marathi Sahayak", layout="centered")
st.title("🇮🇳 SCERT Website Assistant")

# Initialize Engine
@st.cache_resource
def get_engine():
    # Ensure this function connects to your CLOUD database if deployed
    return MarathiRAG()

engine = get_engine()

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # --- COLLAPSIBLE SOURCES FOR HISTORY ---
        if "sources" in message and message["sources"]:
            # Use an expander with a clear label
            with st.expander("🔗 संदर्भ पहा (View Sources)"):
                for url in message["sources"]:
                    st.markdown(f"- [{url}]({url})")

# Handle New Input
if prompt := st.chat_input("तुमचा प्रश्न येथे विचारा..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("विचार करत आहे... (Thinking...)"):
            response_text, sources = engine.generate_answer(
                prompt, 
                chat_history=st.session_state.messages
            )
            
            st.markdown(response_text)
            
            # --- COLLAPSIBLE SOURCES FOR NEW ANSWER ---
            if sources:
                with st.expander("🔗 संदर्भ पहा (View Sources)"):
                    for url in sources:
                        st.markdown(f"- [{url}]({url})")
                        
    # 3. Save Assistant Message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources
    })