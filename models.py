from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, LSTM, Embedding, Input, concatenate, Bidirectional
from keras.models import load_model

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
    filepath="/scratch/gussteen/final_project/checkpoint/atta_lstm.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.compile(optimizer='adam',
                    loss={
                        'word_prediction': 'categorical_crossentropy'
                    },
                    metrics=['accuracy'])

    model.summary()

    history = model.fit([X_before, X_after], y_cat, batch_size=64, epochs=30,
                    callbacks=[csv_logger, earlystopping, checkpoint], validation_split=0.30, verbose=2)
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
    filepath="/scratch/gussteen/final_project/checkpoint/atta_blstm.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.compile(optimizer='adam',
                    loss={
                        'word_prediction': 'categorical_crossentropy'
                    },
                    metrics=['accuracy'])

    model.summary()

    history = model.fit([X_before, X_after], y_cat, batch_size=64, epochs=30,
                    callbacks=[csv_logger, earlystopping, checkpoint], validation_split=0.30, verbose=2)
    model.save('./models/atta_blstm.hdf5')
    return model

def blstm_dropout(X_before, X_after, y_cat, vocab_size, name, load_from=None, dropout=0.2, embed_size = 100, lstm_units = 50, epochs = 40, batch_size=64):
    if load_from: 
        print("Loading model from:", load_from)
        model = load_model(load_from)
    else:
        before_length = X_before.shape[1]
        after_length = X_after.shape[1]
        input_length = before_length + after_length
        cats_length = y_cat.shape[1]

        before_input = Input(shape=(before_length,), name="before_input")
        after_input = Input(shape=(after_length,), name="after_input")

        inputs = concatenate([before_input, after_input])

        embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
        #drop1 = Dropout(dropout)(embed)
        blstm = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout))(embed)
        #drop2 = Dropout(dropout)(blstm)

        # Word prediction softmax
        word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(blstm)

        # This creates a model that includes
        # the Input layer and two Dense layers outputs
        model = Model(inputs=[before_input, after_input], outputs=word_pred)

        model.compile(optimizer='adam',
                        loss={
                            'word_prediction': 'categorical_crossentropy'
                        },
                        metrics=['accuracy'])

    csv_logger = CSVLogger('./results/blstm_{}_log.csv'.format(name), append=True, separator=',')
    # earlystopping = EarlyStopping(monitor='val_loss', patience=2)
    filepath="/scratch/gussteen/final_project/checkpoint/atta_blstm_{}.best.hdf5".format(name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    model.summary()

    history = model.fit([X_before, X_after], y_cat, batch_size=batch_size, epochs=epochs,
                    callbacks=[csv_logger, checkpoint], validation_split=0.30, verbose=2)
    model.save('./models/atta_blstm_{}.hdf5'.format(name))
    return model