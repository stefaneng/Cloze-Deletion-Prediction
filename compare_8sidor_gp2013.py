from utils import *
from models import blstm_dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import pandas as pd
import numpy as np

atta_df = pd.read_csv('/scratch/gussteen/final_project/attasidor.csv')
gp_df = pd.read_csv('/scratch/gussteen/final_project/gp2013_sample.csv')


# Add 8 Sidor sentences
atta_df['word'] = atta_df['word'].astype(str)
all_sents = atta_df.groupby('sent_id')['word'].apply(lambda x: ' '.join(x))

# Add the GP sentences
gp_df['word'] = gp_df['word'].astype(str)
all_sents = all_sents.append(gp_df.groupby('sent_id')['word'].apply(lambda x: ' '.join(x)))

print("Total sentences:", len(all_sents))

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='UNK', filters='–—!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(all_sents)

word_lookup = {v: k for k, v in tokenizer.word_index.items()} 

### Analysis of GP2013
sentence_list = gp_df.groupby('sent_id')['word'].apply(list)
pos_list = gp_df.groupby('sent_id')['pos'].apply(list)

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
        
max_x = len(X_ids)

print("Created {} training examples for GP2013".format(len(y)))
        
X_ids = np.array(X_ids)
X_before = np.array(X_before)
X_after = np.array(X_after)
y_cat = to_categorical(y, num_classes = max_words + 1)

#### MODEL GP2013
blstm_dropout(X_before, X_after, y_cat, max_words + 1, "gp2013_comp", epochs=30, dropout=0.2)

### Analysis of 8 Sidor data
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

# Take the same number of examples as used in the GP2013 test
sample = np.random.choice(len(y), size=max_x, replace=False)
X_ids = np.array(X_ids)[sample]
X_before = np.array(X_before)[sample]
X_after = np.array(X_after)[sample]
y = np.array(y)[sample]
y_cat = to_categorical(y, num_classes = max_words + 1)


#### MODEL 8 Sidor
blstm_dropout(X_before, X_after, y_cat, max_words + 1, "8sidor_comp", epochs=30, dropout=0.2)