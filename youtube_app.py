import streamlit as st
import os
from dotenv import load_dotenv

# Essential LangChain & Google Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- APP CONFIG ---
st.set_page_config(page_title="YouTube Insight Engine", page_icon="📺")
st.title("📺 YouTube Insight Engine")

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Video Settings")
    video_url = st.text_input("Paste YouTube URL:")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- LOGIC: PROCESS VIDEO ---
@st.cache_resource(show_spinner="Analyzing video transcript...")
def process_video(url):
    try:
        # add_video_info=False is the 'magic' fix to avoid "Failed to load" errors
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=False, 
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30
        )
        docs = loader.load()
        
        if not docs:
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Engine Error: {str(e)}")
        return None

# --- MAIN INTERFACE ---
if video_url:
    # Check if user changed the video URL
    if st.session_state.current_video != video_url:
        st.session_state.current_video = video_url
        st.session_state.chat_history = [] # Wipe chat for new video
        st.cache_resource.clear() # Clear cache for new vector store

    retriever = process_video(video_url)
    
    if retriever:
        st.video(video_url)
        st.divider()

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # AI Chain Setup
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        
        system_prompt = (
            "You are a video expert. Use the context to answer. "
            "Context has 'start_timestamp' metadata. "
            "ALWAYS cite the time as [at MM:SS].\n\nCONTEXT: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Chat Input
        query = st.chat_input("Ask something about the video...")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
else:
    st.info("Paste a link to begin!")