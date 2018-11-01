from utils import *
from models import blstm_dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import pandas as pd
import numpy as np

atta_df = pd.read_csv('/scratch/gussteen/final_project/attasidor.csv')

atta_df['word'] = atta_df['word'].astype(str)
atta_all_sents = atta_df.groupby('sent_id')['word'].apply(lambda x: ' '.join(x))

print("Total sentences:", len(atta_all_sents))

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='UNK', filters='–—!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(atta_all_sents)

word_lookup = {v: k for k, v in tokenizer.word_index.items()} 

sentence_list = atta_df.groupby('sent_id')['word'].apply(list)
pos_list = atta_df.groupby('sent_id')['pos'].apply(list)

X_ids = []
X_before = []
X_after = []
y = []
for words, w_pos, s_id in zip(list(sentence_list), list(pos_list), list(sentence_list.index)):
    for ex_id, before, after, w in create_training_example(words, w_pos, s_id, tokenizer):
        X_ids.append(ex_id)
        X_before.append(before)
        X_after.append(after)
        y.append(w)

print("Created {} training examples".format(len(y)))
        
X_ids = np.array(X_ids)
X_before = np.array(X_before)
X_after = np.array(X_after)
y_cat = to_categorical(y, num_classes = max_words + 1)


#### MODEL
blstm_dropout(X_before, X_after, y_cat, max_words + 1, "dropout4", epochs=25, dropout=0.4)
blstm_dropout(X_before, X_after, y_cat, max_words + 1, "dropout6", epochs=25, dropout=0.6)
blstm_dropout(X_before, X_after, y_cat, max_words + 1, "dropout8", epochs=25, dropout=0.8)