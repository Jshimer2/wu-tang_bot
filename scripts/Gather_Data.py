#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:01:44 2020

@author: jordanshimer
"""
import pandas as pd
import lyricsgenius as genius
import re

api_key = 'PN75lo_JAlqJKXKxHG3ZWtnuii4EAeA-w2XWjSBgtE6MtB2PgSNPOTUDAIf1-E8p'
save_files = True

def get_538_tweets(nrows = None):
    files = []
    for i in range(1,14): # There are 13 files, numbered 1-13
        print('Loading file {}'.format(i))
        # Insert the number into the url, then read the file to a list
        url = 'https://raw.githubusercontent.com/fivethirtyeight/russian-troll-tweets/master/IRAhandle_tweets_{}.csv'.format(i) 
        files.append(pd.read_csv(url, nrows = nrows).content)
    df_tweets = pd.DataFrame(pd.concat(files).reset_index(drop = True)).iloc[:10000]
    
    return(df_tweets)
    
def get_artist_tweets(artist_name = 'Wu-Tang Clan',
                      api_key = None, 
                      max_songs = 100):
    if not api_key:
        raise ValueError('This script requires a Genius.com API Key. Please go to https://docs.genius.com/#/getting-started-h1 to get a key (They\'re free!)')
    
    # Any errors here, report to https://github.com/johnwmillr/LyricsGenius
    # or check docs.genius.com
    api = genius.Genius(api_key)
    artist = api.search_artist(artist_name, max_songs = max_songs) 
    
    artist_lyrics = []
    for song in artist.songs:
        artist_lyrics.append(song.lyrics)
    df_lyrics = pd.DataFrame(columns = ['content'], data = artist_lyrics)
    
    return(df_lyrics)
    
def clean_lyrics(df_lyrics, column = 'content'):
    
    # Remove anything in quotes or brackets
    brackets = re.compile(r'\[.*\]')
    double_space = re.compile('\s+')
    def apply_regexes(raw_string):
        temp_string = brackets.sub('', raw_string)
        temp_string = double_space.sub(' ', temp_string)
        
        return(temp_string)
        
    df_lyrics[column] = df_lyrics[column].apply(apply_regexes)
    df_lyrics[column] = df_lyrics[column].str.replace('"', '')
    df_lyrics[column] = df_lyrics[column].str.replace('\'', '')
    
    df_lyrics['source'] = 'lyrics'
    
    return(df_lyrics)
    
def clean_tweets(df_tweets, column = 'content'):
    links = re.compile(r'http\S+')
    double_space = re.compile('\s+')
    mentions = re.compile('@\S+')
    def apply_regexes(raw_string):
        temp_string = links.sub('', raw_string)
        temp_string = mentions.sub(' ', temp_string)
        temp_string = double_space.sub(' ', temp_string)

        return(temp_string)
        
    df_tweets[column] = df_tweets[column].apply(apply_regexes)
    df_tweets[column] = df_tweets[column].str.replace('#', '')
    df_tweets[column] = df_tweets[column].str.replace('"', '')
    df_tweets[column] = df_tweets[column].str.replace('\'', '')
    
    df_tweets['source'] = 'tweets'
    
    return(df_tweets)

if __name__ == '__main__':
    df_tweets = get_538_tweets()
    df_tweets = clean_tweets(df_tweets)
    df_tweets.to_csv('../data/tweets.csv')

    df_lyrics = get_artist_tweets(api_key = api_key)
    df_lyrics = clean_lyrics(df_lyrics)
    df_lyrics.to_csv('../data/lyrics.csv')
        