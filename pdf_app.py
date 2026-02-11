import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- APP CONFIG ---
st.set_page_config(page_title="Universal PDF Analyzer", page_icon="📄")
st.title("📄 Universal PDF Analyzer")

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # To display in the UI
if "report_log" not in st.session_state:
    st.session_state.report_log = ["--- PDF ANALYSIS REPORT ---"]  # For the download file

# --- SIDEBAR: CONTROLS & EXPORT ---
with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Drag and drop your PDF", type="pdf")
    
    st.divider()
    
    st.header("2. Controls")
    # CLEAR CHAT BUTTON
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.report_log = ["--- PDF ANALYSIS REPORT ---"]
        st.rerun()

    st.divider()
    
    st.header("3. Export")
    # Prepare the download data by joining the log list
    report_data = "\n\n".join(st.session_state.report_log)
    
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_data,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

# --- PROCESSING LOGIC ---
@st.cache_resource(show_spinner="Analyzing document...")
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.getbuffer())
        file_path = tf.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# --- MAIN INTERFACE ---
if uploaded_file:
    retriever = process_pdf(uploaded_file)
    
    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Setup the Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    system_prompt = "You are an expert assistant. Use the following context to answer.\n\nCONTEXT: {context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # User Input
    query = st.chat_input("Ask a question about your document...")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            st.write(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Update the report log for downloading
        st.session_state.report_log.append(f"USER: {query}\nAI: {answer}")
        
        # Force a rerun to update the download button data in the sidebar
        st.rerun()

else:
    st.info("Please upload a PDF to begin analysis.")