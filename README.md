Autonomous Multi-Modal RAG ReAct Pipeline

Overview
A production-grade Agentic Intelligence system that transforms unstructured data into validated business insights. This pipeline leverages a Reasoning + Acting (ReAct) framework to autonomously decide when to search internal documents, browse the live web, or perform precise mathematical calculations.

Key Architectural Pillars

1. Autonomous ReAct Logic
Unlike standard RAG, this system uses a Reasoning Loop. The Agent analyzes the user's intent and dynamically selects the best tool for the job:
- Internal PDF Search: Semantic retrieval from uploaded documents.
- Web Search (SerpApi): Real-time market data and competitor grounding.
- Table Extractor (pdfplumber): Precise grid-data extraction from PDF rows and columns.
- Math Tool: A logic-gate for verified financial calculations like ROI or CAGR.

2. Multi-Modal Data Ingestion
- Unstructured Text: Recursive character splitting for semantic context.
- Structured Tables: Dedicated parsing to prevent the flattening of numerical data.
- Real-time Data: Live web-scraping bridge for up-to-the-minute accuracy.

3. Enterprise-Ready Stack
- Backend: FastAPI microservice architecture with asynchronous endpoints.
- Containerization: Fully Dockerized (Compose) for consistent deployment.
- Vector Engine: ChromaDB for high-dimensional embedding storage.

Tech Stack

- Orchestration: LangChain (AgentExecuter, ReAct)
- Brain (LLM): Google Gemini 3 Flash
- Embeddings: Google text-embedding-004
- Vector Store: ChromaDB
- API Layer: FastAPI
- Frontend: Streamlit
- Parsing: pdfplumber and PyPDF
- Live Search: SerpApi

Installation and Setup

1. Clone and Configure
git clone https://github.com/MEHUL-NS/Agentic-RAG-Pipeline-for-Document-Intelligence.git
cd Agentic-RAG-Pipeline-for-Document-Intelligence

2. Environment Variables
Create a .env file with your API keys:
GOOGLE_API_KEY=your_gemini_key
SERPAPI_API_KEY=your_serpapi_key

3. Run via Docker (Recommended)
This launches both the FastAPI backend and Streamlit frontend in a networked environment.
docker-compose up --build

4. Manual Setup (Alternative)
Terminal 1 (Backend): uvicorn main:app --reload
Terminal 2 (Frontend): streamlit run app.py

Example Agentic Workflow

User Prompt: Find the profit margin on page 4 of the PDF and compare it to the current market average for AI startups.

- Thought: I need to find the profit margin in the document first.
- Action: Table_Tool(page="4") Result: 18%
- Thought: Now I need the current market average from the web.
- Action: Web_Search("average profit margin AI startups 2026") Result: 12%
- Thought: I should calculate the difference.
- Action: Math_Tool("18 - 12") Result: 6%
- Final Response: The internal profit margin is 18%, which is 6% higher than the current market average of 12%.

