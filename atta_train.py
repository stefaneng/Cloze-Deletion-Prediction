import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, CSVLogger
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import itertools
import numpy as np

from keras.models import Model
from keras.layers import Dense, LSTM, Embedding, Input, Dropout, concatenate, Bidirectional

def create_training_example(sentence, pos_tags, sent_id, tokenizer, cloze_pos=['NN', 'VB', 'JJ'], window_size=3):
    for i, word in enumerate(sentence):   
        if pos_tags[i] in cloze_pos:
            before = sentence[max(i-window_size, 0):i]
            after = sentence[i+1:i+window_size]
            seqs = tokenizer.texts_to_sequences([before, after, [word]])
            # Skip words that are out of vocabulary
            # We don't want OOV words as any of the predicting values
            # 1 is OOV index
            if 1 in itertools.chain(*seqs):
                continue
            seqs[0] = pad_sequences([seqs[0]], maxlen=window_size, padding='pre', truncating='pre')[0]
            seqs[1] = pad_sequences([seqs[1]], maxlen=window_size, padding='post', truncating='post')[0]
            
            yield sent_id, seqs[0], seqs[1], seqs[2]

atta_df = pd.read_csv('/scratch/gussteen/final_project/attasidor_sample.csv')

atta_df['word'] = atta_df['word'].astype(str)
atta_all_sents = atta_df.groupby('sent_id')['word'].apply(lambda x: ' '.join(x))

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

X_ids = np.array(X_ids)
X_before = np.array(X_before)
X_after = np.array(X_after)
y_cat = to_categorical(y, num_classes = max_words + 1)


#### MODEL
vocab_size = max_words + 1
embed_size = 100

before_length = X_before.shape[1]
after_length = X_after.shape[1]
input_length = before_length + after_length
cats_length = y_cat.shape[1]

## Model 1
def simple_lstm():
    before_input = Input(shape=(before_length,), name="before_input")
    after_input = Input(shape=(after_length,), name="after_input")

    inputs = concatenate([before_input, after_input])

    embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
    lstm = LSTM(50, dropout=0.1)(embed)

    # Word prediction softmax
    word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(lstm)

    # This creates a model that includes
    # the Input layer and two Dense layers outputs
    model = Model(inputs=[before_input, after_input], outputs=word_pred)

    csv_logger = CSVLogger('./results/atta_lstm_log.csv', append=True, separator=',')
    earlystopping = EarlyStopping(monitor='loss', patience=2)

    model.compile(optimizer='adam',
                    loss={
                        'word_prediction': 'categorical_crossentropy'
                    },
                    metrics=['accuracy'])

    model.summary()

    history = model.fit([X_before, X_after], y_cat, batch_size=32, epochs=40,
                    callbacks=[csv_logger, earlystopping])
    model.save('./models/atta_lstm.hdf5')
    return model

def simple_blstm():
    before_input = Input(shape=(before_length,), name="before_input")
    after_input = Input(shape=(after_length,), name="after_input")

    inputs = concatenate([before_input, after_input])

    embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
    blstm = Bidirectional(LSTM(50, dropout=0.1))(embed)

    # Word prediction softmax
    word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(blstm)

    # This creates a model that includes
    # the Input layer and two Dense layers outputs
    model = Model(inputs=[before_input, after_input], outputs=word_pred)

    csv_logger = CSVLogger('./results/atta_blstm_log.csv', append=True, separator=',')
    earlystopping = EarlyStopping(monitor='loss', patience=2)

    model.compile(optimizer='adam',
                    loss={
                        'word_prediction': 'categorical_crossentropy'
                    },
                    metrics=['accuracy'])

    model.summary()

    history = model.fit([X_before, X_after], y_cat, batch_size=32, epochs=40,
                    callbacks=[csv_logger, earlystopping])
    model.save('./models/atta_blstm.hdf5')
    return model

simple_blstm()