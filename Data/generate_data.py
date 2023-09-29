import os
#import h5py
from urllib.request import urlretrieve
import pandas as pd

#downloading latest file from link provided on concept net github
if not os.path.isfile('mini.h5'):
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5'
    urlretrieve(conceptnet_url, 'mini.h5')
     
df = pd.read_hdf('mini.h5')  

#Filter on only english words and save that as a separate file
englishWords = []
englishEmbeddings = []

for index, row in df.iterrows():
    word, embedding = index[6:], row.values
    if index.startswith('/c/en'):
        englishWords.append(word)
        englishEmbeddings.append(embedding)  
        
englishdf = pd.DataFrame(englishEmbeddings, index=englishWords)
englishdf.to_pickle('./englishWords.pkl')

#Testing the saved file file
#df = pd.read_pickle('./englishWords.pkl')