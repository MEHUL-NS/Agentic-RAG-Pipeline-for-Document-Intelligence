import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🤖 Autonomous RAG Agent")
st.markdown("---")

# 2. Sidebar for PDF Ingestion
with st.sidebar:
    st.header("1. Upload Knowledge")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Index Document"):
        if uploaded_file:
            with st.spinner("Indexing PDF..."):
                files = {"file": uploaded_file.getvalue()}
                # Sending the file to your FastAPI backend
                response = requests.post("http://127.0.0.1:8000/ingest", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
                if response.status_code == 200:
                    st.success("Document Indexed Successfully!")
                else:
                    st.error(f"Error: {response.text}")
        else:
            st.warning("Please upload a file first.")

# 3. Chat Interface
st.header("2. Ask the Agent")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about ROI, market data, or calculations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query", 
                    json={"question": prompt}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("The agent encountered an error.")
            except Exception as e:
                st.error(f"Connection failed: {e}")