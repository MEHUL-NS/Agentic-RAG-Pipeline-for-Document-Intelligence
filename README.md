Overview: A high-performance Retrieval-Augmented Generation (RAG) system designed to transform unstructured PDF data into actionable insights using LangChain and Google Gemini.

## Key Features:
 - Advanced RAG Architecture: Utilizes Recursive Character Text Splitting and ChromaDB vector embeddings for high-precision context retrieval.
 - Contextual Awareness: Implemented session-state management to maintain chat history, ensuring "transparency and alignment" in multi-turn conversations.
 - Scalable Backend: Designed with a modular structure ready for migration to FastAPI microservices.
 - Exportable Insights: Built-in reporting tool to download analysis logs for business stakeholders.

## How It Works 
 - Ingestion: PDF loading via PyPDFLoader.
 - Chunking: Splitting documents into 1000-character segments with overlap to preserve context.
 - Retrieval: Hybrid-ready search using GoogleGenerativeAIEmbeddings.
 - Generation: Response synthesis using a ChatPromptTemplate for expert-level assistant persona.

## Tech Stack
 - LLM Orchestration:LangChain (Chains, Prompt Templates, Memory)
 - Generative Model: Google Gemini (Gemini-1.5-Flash-Lite)
 - Vector Store: ChromaDB (Vector Embeddings & Similarity Search)
 - Embeddings: Google Generative AI Embeddings
 - Frontend/UI: Streamlit
 - Environment & Tools: Python, Dotenv, PyPDF

 ## Installation & Setup
1. Clone the Repository:
   ```bash
   git clone [https://github.com/MEHUL-NS/Agentic-RAG-Pipeline-for-Document-Intelligence.git](https://github.com/YOUR_USERNAME/Agentic-RAG-Pipeline-for-Document-Intelligence.git)
   cd Agentic-RAG-Pipeline-for-Document-Intelligence
   
2. Create a Virtual Environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Environmental Configurations:
   GOOGLE_API_KEY=your_actual_key_here

5. Run the Application:
   streamlit run app.py





   
   git clone [https://github.com/YOUR_USERNAME/Agentic-RAG-Pipeline-for-Document-Intelligence.git](https://github.com/YOUR_USERNAME/Agentic-RAG-Pipeline-for-Document-Intelligence.git)
   cd Agentic-RAG-Pipeline-for-Document-Intelligence
