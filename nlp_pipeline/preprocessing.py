from tensorflow.keras.preprocessing.sequence import pad_sequences
from nlp_pipeline.value import MAX_LENGTH, PADDING_TYPE, TRUNC_TYPE
import pickle

def tokenization(payload):
    # Define tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
      tokenizer = pickle.load(handle)
    
    input_sequence = tokenizer.texts_to_sequences(payload)
    input_padded = pad_sequences(input_sequence, padding = PADDING_TYPE, maxlen= MAX_LENGTH, truncating = TRUNC_TYPE)
    
    return input_padded