import numpy as np
import pandas as pd
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import random
from nltk.stem import WordNetLemmatizer
import wordfreq



""" To Do: Implement stemming (Done), removing numbers (Done), removing words with multiple ### at the start (Done), rejecting common words like (a, the, and etc.) (Done) """
""" To Do: Replace stemming with lemmatization (Done) """
""" To Do: Think of ways to filter/reject guesses where both words are basically the same (e.g word1 = apple and word2 = apples), stemming is probably a good way to do this (Done via lemmatisation) """
""" TO Do: Implement a basic front-end chat room to play with the bot"""
""" TO Do: Host on a server"""


lemmatizer = WordNetLemmatizer()

def loadData():
    df = pd.read_pickle('./Data/englishWords.pkl')
    return df

def cleanUp(word):
    word = lemmatizer.lemmatize("".join(c for c in word if c.isalpha()))
    return word

def pickRandomWord(df):
    freq = 0
    #do not pick words that are too frequent or too infrequent
    while freq <= 1 or freq >= 5:
        randWord = cleanUp(random.choice(df.index))
        freq = wordfreq.zipf_frequency(randWord, 'en', 'large')
    return randWord

#looks lime lemmatization can sometimes produce a non-ideal result (e.g returning 'courted' as an answer), but the overall result is much better than using stemming
#as stemming was consistenly returning non-sensical words
def findSimilarWord(word1, word2, df):
    #no need to clean up word1, since that is always done in the return of this function/pickRandomWord (word1 is the bot's guess)
    word2 = cleanUp(word2)
    combinedEmbedding = np.array(df.loc[word1]) + np.array(df.loc[word2])
    
    similarities = [(0,'dummy')]
    curWords = set()
    curWords.add('dummy')
    for word, embedding in df.iterrows():
        #Do not use the same words as the inputs or words that are non sensical (e.g words that have # at the start), also reject infrequent/frequent words or words that are similar to ones in our 
        #current top 10 (e.g 'court' and 'courts' should not both be present in our top 10)
        word = cleanUp(word)
        freq = wordfreq.zipf_frequency(word, 'en', 'large')
        if (word in (word1, word2)
            or word in curWords
            or freq <=1 or freq >=5):
            continue
        
        embedding = np.array(embedding.values)      
        cosineSim = cosine_similarity(combinedEmbedding.reshape(1, -1), embedding.reshape(1, -1))
        if cosineSim > similarities[0][0]:
            curWords.add(word)
            heapq.heappush(similarities, (cosineSim, word))
            
            if len(similarities) > 10:
                popped = heapq.heappop(similarities)
                curWords.remove(popped[1])
            
    return random.choice(list(curWords)), curWords

df = loadData()

word1, word2 = pickRandomWord(df), pickRandomWord(df)
print(word1, word2)

out, words = findSimilarWord("tennis", "court", df)
print(out, words)
        
        
    
    