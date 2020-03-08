#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:00:07 2020

@author: jordanshimer
"""

import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical


lyrics_path = '../data/lyrics.csv'
tweets_path = '../data/tweets.csv'

def load_data(lyrics_path, tweets_path):
    df_lyrics = pd.read_csv(lyrics_path)    
    df_tweets = pd.read_csv(tweets_path)
    return(df_lyrics, df_tweets)

def add_start_end_tokens(string, start = 'STARTTOKEN', end = 'ENDTOKEN'):
    return(' '.join([start, string, end]))
    
def prepare_dfs(df_lyrics, df_tweets, column = 'content'):
    df = pd.concat([df_lyrics, df_tweets]).reset_index(drop = True).sample(5000)
    
    df.dropna(how = 'any', inplace = True)
    
    df.content.apply(add_start_end_tokens)
    
    data = list(df.content)
    
    # integer encode sequences of words
    tokenizer = Tokenizer(num_words = 5000)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)   
    
    input_texts = []
    output_word = []
    for obs in sequences:
        for target_word in range(1,len(obs)):
            input_texts.append(np.array(obs[:target_word]))
            output_word.append(obs[target_word])
    
    X = pad_sequences(input_texts)
    y = to_categorical(output_word)
            
            

    with open('../model/tokenizer.pickle', 'wb') as fp:
        pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    return(X, y)

    
if __name__ == '__main__':
    df_lyrics, df_tweets = load_data(lyrics_path, tweets_path)
    X, y = prepare_dfs(df_lyrics, df_tweets)
    
    np.save('../data/arrays/target.npy', X)
    np.save('../data/arrays/input.npy', y)
    
    
    
