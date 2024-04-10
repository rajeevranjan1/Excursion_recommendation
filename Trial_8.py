

# In[1]:


import pandas as pd


# In[2]:


# df=pd.read_csv('generated_dataset.csv')
df=pd.read_csv('input_output.csv')


# In[3]:


df.head(1)


# In[4]:


input_texts=df['input_text'].to_list()
target_texts=df['target_text'].to_list()


# In[5]:


target_texts = ['\t ' + text + ' \n' for text in target_texts]


# In[6]:


# # Sample dataset
# input_texts = ['I am a student.', 'This is a pen.', 'You have a book.']
# target_texts = ['Je suis étudiant.', 'Ceci est un stylo.', 'Tu as un livre.']

# # Add start and end tokens to target texts
# target_texts = ['\t ' + text + ' \n' for text in target_texts]


# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize input texts
input_tokenizer = Tokenizer(char_level=False)
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_sequences = pad_sequences(input_sequences, padding='post')

# Tokenize target texts
target_tokenizer = Tokenizer(char_level=False, filters='')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_sequences = pad_sequences(target_sequences, padding='post')

# Vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Maximum sequence lengths
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in target_sequences])


# In[8]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(128, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & summarize
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[9]:


import numpy as np

decoder_input_data = target_sequences[:, :-1]
decoder_target_data = target_sequences[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

model.fit(
    [input_sequences, decoder_input_data], decoder_target_data,
    #batch_size=64,
    epochs=100,
    validation_split=0.2
)


# In[10]:


# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# In[11]:


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
        decoded_sentence += sampled_char+' '
        
        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence


# In[14]:


input_text = "Excursion details -1st Stage Separator train B oil level went more than high high limit 2 to 3 time i.e. more than 89.5% after startup.Level control valve was 100% open  working as per level control philosphy  but more flow can be achieved through"
decoded_sentence = decode_sequence(input_text)
print("Input:", input_text)
print("Output:", decoded_sentence)


# In[ ]:




