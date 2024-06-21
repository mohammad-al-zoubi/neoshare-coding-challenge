import uvicorn
from utils import CLASSES_DICT
from fastapi import FastAPI, HTTPException
from embeddings_classifier.inference import classify_text as classify_text_linear
from embeddings_classifier.embeddings_generator import cohere_embeddings


app = FastAPI()


@app.get("/predict/")
async def predict(text: str, mode: str = 'linear'):
    try:
        if mode == 'linear':
            embeddings = cohere_embeddings([text])
            result = classify_text_linear(embeddings[0])
            result = CLASSES_DICT[result]
            return {"class": result}
        elif mode == 'llm':
            raise HTTPException(status_code=400, detail="LLM mode not implemented yet.")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server.__main__:app", host="0.0.0.0", port=8000, reload=True)
