from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from src.embedding import upsert_chunks, get_or_create_index
from src.ingestion import ingest
from src.retrieval import search_vector_db
from src.reranker import search_vector_db_reranker
from src.generation import generate_answer


app = FastAPI()


## creating request and response model class 

class IngestRequest(BaseModel):
    file_path : str

    @field_validator('file_path')
    @classmethod
    def normalize_path(cls, v):
        normalized = v.replace("\\", "/")
        return normalized

class IngestResponse(BaseModel):
    source_doc : str 
    chunks : int
    message : str


@app.get("/")
def health():
    return {"status" : "okay"}



@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request : IngestRequest):
    
    try:
        file_path = Path(request.file_path)
        ready_embed_chunks = ingest(request.file_path)
        upserted_chunks = upsert_chunks(ready_embed_chunks)
        return IngestResponse(
            source_doc=request.file_path,
            chunks=upserted_chunks,
            message="Document ingested successfully",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File Not Found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    