import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Core AI & Search Tools
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper # <--- New Web Search Utility

load_dotenv()

app = FastAPI(
    title="Autonomous Document & Market Intelligence API",
    description="RAG + Table Extraction + Math + Real-time Web Search",
    version="3.0.0"
)

# --- GLOBAL STATE ---
state = {"retriever": None, "file_path": None}

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- TOOL DEFINITIONS ---

def document_search_tool(query: str):
    """Searches the internal PDF for facts and context."""
    if not state["retriever"]: return "Error: No document indexed."
    docs = state["retriever"].invoke(query)
    return "\n\n".join([d.page_content for d in docs])

def table_extraction_tool(page_number: str):
    """Extracts raw table data from the PDF for ROI analysis."""
    if not state["file_path"]: return "Error: No file found."
    try:
        page_idx = int("".join(filter(str.isdigit, page_number))) - 1
        with pdfplumber.open(state["file_path"]) as pdf:
            table = pdf.pages[page_idx].extract_table()
            return "\n".join([" | ".join([str(i) if i else "" for i in row]) for row in table])
    except: return "Table extraction failed."

def calculator_tool(expression: str):
    """Performs exact math like percentages or ROI totals."""
    try: return f"Result: {eval(''.join(c for c in expression if c in '0123456789+-*/(). '))}"
    except: return "Math error."

# Initialize Web Search Utility
search = SerpAPIWrapper()

# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        state["file_path"] = file_path

        loader = PyPDFLoader(file_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
        vector_db = Chroma.from_documents(documents=chunks, embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))
        state["retriever"] = vector_db.as_retriever()
        
        return {"status": "success", "message": f"{file.filename} is active."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, thinking_budget=0, include_thought=False)

        agent_tools = [
            Tool(name="Internal_PDF_Search", func=document_search_tool, 
                 description="Use this for info inside the uploaded document."),
            Tool(name="Web_Search", func=search.run, 
                 description="Use this for real-time market data, news, or competitor info."),
            Tool(name="Table_Extractor", func=table_extraction_tool, 
                 description="Use for tables. Input: page number."),
            Tool(name="Calculator", func=calculator_tool, 
                 description="Use for precise math.")
        ]

        executor = AgentExecutor(
            agent=create_react_agent(llm, agent_tools, hub.pull("hwchase17/react")),
            tools=agent_tools, verbose=True, handle_parsing_errors=True, max_iterations=6
        )

        result = executor.invoke({"input": request.question})
        return QueryResponse(answer=result["output"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)