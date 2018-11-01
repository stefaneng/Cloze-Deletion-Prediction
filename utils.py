import itertools
from keras.preprocessing.sequence import pad_sequences

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