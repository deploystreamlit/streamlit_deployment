from fastapi import FastAPI
from model_deployment.routers.prediction import router as prediction_router
import uvicorn
from dotenv import load_dotenv, find_dotenv

env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)

app = FastAPI()

app.include_router(prediction_router, prefix="/predictions")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)