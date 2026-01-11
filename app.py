# app.py
import streamlit as st
from src.rag_engine import MarathiRAG

# Page Config
st.set_page_config(page_title="SCERT Marathi Sahayak", layout="centered")
st.title("🇮🇳 SCERT Website Assistant")

# Initialize RAG Engine (Cached so it doesn't reload on every click)
@st.cache_resource
def get_engine():
    return MarathiRAG()

engine = get_engine()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there were sources saved with this message, show them
        if "sources" in message and message["sources"]:
            st.caption("🔗 संदर्भ (Sources):")
            for url in message["sources"]:
                st.markdown(f"- [{url}]({url})")

# User Input
if prompt := st.chat_input("तुमचा प्रश्न येथे विचारा... (Ask your question here)"):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("शोधत आहे... (Searching...)"):
            # Note: Ensure you are passing 'chat_history' if your engine expects it
            response_text, sources = engine.generate_answer(
                prompt
            )
            
            st.markdown(response_text)
            
            # --- COLLAPSIBLE SOURCES ---
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