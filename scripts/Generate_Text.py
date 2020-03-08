#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:28:35 2020

@author: jordanshimer
"""
import pickle
from keras.models import load_model
import numpy as np

with open('../model/tokenizer.pickle', 'rb') as fp:
    tokenizer = pickle.load(fp)

model = load_model('../model/model')

vocab = dict(tokenizer.word_index)
token_mapping = {vocab[x]:x for x in vocab}

seed_text = 'hello trump'
def predict_text(seed_text, model, tokenizer, index_to_word):   
    seed_text = ' '.join(['STARTTOKEN', seed_text])
    
    # Take the phrase and tokenize it
    new_input = tokenizer.texts_to_sequences([seed_text])
    
    # How long is the input and how long is the output vector
    len_input = len(new_input[0])
    pad_length = model.output.shape[1] - len(new_input[0])
    
    new_input[0].extend([0.0 for _ in range(pad_length)])

    for word in range(len_input, pad_length):
        pred = np.argmax(model.predict(new_input[0][:word])[word-1])
        new_input[0][word] = pred    
    
    output = [index_to_word[i] for i in new_input[0] if i != 0]
    
    output = ' '.join(output)
    
    output = output.replace('STARTTOKEN ', '')
    output = output.split('ENDTOKEN')[0]
    
    return(output)

if __name__ == '__main__':
    out = predict_text(seed_text, model, tokenizer, token_mapping)
    print(out)