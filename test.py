from nlp_pipeline.preprocessing import tokenization
from nlp_pipeline.inference import predict

def predict(payload):
    vector = tokenization(payload)
    prediction = predict(vector)
    
    response = {
        'Result': prediction
    }
    
    return response

predict("BAGUS")