from fastapi import FastAPI
from config import app_config
from payload import Payload
from nlp_pipeline.preprocessing import tokenization
from nlp_pipeline.inference import predict

app = FastAPI(**app_config)

@app.post('/spam-classifier/')
async def predict(payload: Payload):
    vector = tokenization(payload)
    prediction = predict(vector)
    
    response = {
        'Result': prediction
    }
    
    return response