import streamlit as st
import requests

# Page UI configurations
st.set_page_config(page_title="RAG Document Analyzer", page_icon="📄")
st.title("📄 Local RAG Document Analyzer")
st.subheader("Query your knowledge base via a decoupled full-stack architecture")

FASTAPI_URL = "http://127.0.0.1:8000/query"

# Initialize a chat session state history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_prompt := st.chat_input("Ask a question about your documents..."):
    
    # Render user query instantly
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Query the FastAPI backend service
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching database and generating synthesis...")
        
        try:
            payload = {"prompt": user_prompt}
            response = requests.post(FASTAPI_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
                message_placeholder.markdown(answer)
                # Store the successful response in session history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Backend Error: {response.json().get('detail', 'Unknown error')}"
                message_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Could not connect to FastAPI server. Ensure it is running on port 8000.")