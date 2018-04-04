import numpy as np 
import pandas as pd 

from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, Flatten, AveragePooling1D
from keras.layers import LSTM, Bidirectional, Dropout, TimeDistributed
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from input import load_data

print('------- LOADING DATA... -------')
x, y, vocabulary, vocabularyInv = load_data()

print('------- SPLITING DATA... -------')
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1] # 56
max_features = len(vocabularyInv) 
xTrain = sequence.pad_sequences(xTrain, maxlen=sequence_length)
xTest = sequence.pad_sequences(xTest, maxlen=sequence_length)

epochs = 2
batch_size = 30

model = Sequential()
model.add(Embedding(max_features, 128, input_length=sequence_length))
model.add(Bidirectional(LSTM(64, recurrent_dropout=0.2, return_sequences=True)))
model.add(TimeDistributed(Dense(1)))
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, validation_data=(xTest, yTest), callbacks=callbacks_list)

score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])