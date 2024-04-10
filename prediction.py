from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

df=pd.read_csv('generated_dataset.csv',encoding='utf-8')

input_texts=df['input_text'].to_list()
target_texts=df['target_text'].to_list()

target_texts = ['\t ' + text + ' \n' for text in target_texts]
# Tokenize input texts
input_tokenizer = Tokenizer(char_level=True)
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_sequences = pad_sequences(input_sequences, padding='post')

# Tokenize target texts
target_tokenizer = Tokenizer(char_level=True, filters='')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_sequences = pad_sequences(target_sequences, padding='post')

# Vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Maximum sequence lengths
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in target_sequences])

# Load a model from the SavedModel format
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

def decode_sequence(input_text):
    # Tokenize the input text
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1 with only the start character.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character code.
    # Note: "\t" is used as the "start sequence" character for the targets, and "\n" as the "end sequence" character.
    target_seq[0, 0] = target_tokenizer.word_index['\t']
    
    # Looping variable to keep track of when to stop the loop (when we hit the max length or find the stop character)
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += sampled_char
        
        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence

input_text = "System temperature increased to 85 degC when pressure is 60 bar and valve is 70% open."
decoded_sentence = decode_sequence(input_text)
print("Input:", input_text)
print("Output:", decoded_sentence)