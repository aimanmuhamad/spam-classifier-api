from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nlp_pipeline.value import MAX_LENGTH, VOCAB_SIZE, OOV_TOK, PADDING_TYPE, TRUNC_TYPE

def tokenization(payload : str):
    tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = OOV_TOK)
    input_sequences = tokenizer.texts_to_sequences(payload)
    input_padded = pad_sequences(input_sequences, padding = PADDING_TYPE, maxlen= MAX_LENGTH, truncating = TRUNC_TYPE)
    
    return input_padded