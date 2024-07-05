from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Asumimos que tienes un objeto 'qa' ya configurado
from rag import rag_with_llama2 as qa

app = FastAPI()

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/query", response_model=Answer)
async def query(query: Query):
    try:
        # Asumimos que 'qa' tiene un método 'run' que toma una pregunta y devuelve una respuesta
        result = qa.run(query.question)
        return Answer(answer=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)