from fastapi import FastAPI
from config import app_config
from payloads import Payload

app = FastAPI(**app_config)

@app.post('/spam-classifier/')
async def predict(payload: Payload):
    a = payload
    
    return a