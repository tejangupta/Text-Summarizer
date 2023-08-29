from fastapi import FastAPI
from starlette.responses import RedirectResponse
from textSummarizer.pipeline.prediction import PredictionPipeline
import uvicorn

text: str = "What is Text Summarization?"

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)

        return text
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
