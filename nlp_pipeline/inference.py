from keras.models import load_model

def predict(input):
    
    predicts = {1.0: "Non-Spam",
                2.0: "Spam"
               }
    model = load_model('model/email_classification.h5')
    
    prediction = model.predict(input)
    float_index = float(prediction[0][0])
    
    return float_index