# The attention layer must use theano backend
#import os
#os.environ["KERAS_BACKEND"]="theano"

import numpy as np
import pdb

from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from AttentionLayer import Attention
import utils

xtra, ytra, xval, yval = utils.readMemmap()

main_input = Input(shape= (xtra.shape[1],), dtype='float32')

embedded_input = Embedding(27,27,input_length=452)(main_input)
embedded_input = Dropout(0.2)(embedded_input)
x1 = Conv1D(1, 16, strides=1, activation='relu')(embedded_input)
x1 = MaxPooling1D(pool_size=4, strides=1)(x1)
x1 = Dropout(0.2)(x1)
x1 = GRU(128, return_sequences=True, activation='tanh')(x1)
x1= Dropout(0.2)(x1)
x1 = GRU(64, return_sequences=True, activation='tanh')(x1)
x1= Dropout(0.2)(x1)
x1 = GRU(32, return_sequences=True, activation='tanh')(x1)
x1 = Attention()(x1)

embedded_input2 = Embedding(27,27,input_length=452)(main_input)
embedded_input2 = Dropout(0.2)(embedded_input2)
x2 = Conv1D(1, 16, strides=1, activation='relu')(embedded_input2)
x2 = MaxPooling1D(pool_size=4, strides=4)(x2)
x2 = Dropout(0.2)(x2)
x2 = GRU(128, return_sequences=True, activation='tanh')(x2)
x2= Dropout(0.2)(x2)
x2 = GRU(64, return_sequences=True, activation='tanh')(x2)
x2= Dropout(0.2)(x2)
x2 = GRU(32, return_sequences=True, activation='tanh')(x2)
x2 = Attention()(x2)
x2 = Activation(activation='softmax')(x2)

x12 = Multiply()([x1,x2])

embedded_input3 = Embedding(27,27,input_length=452)(main_input)
embedded_input3 = Dropout(0.2)(embedded_input3)
x3 = Conv1D(1, 16, strides=1, activation='relu')(embedded_input3)
x3 = MaxPooling1D(pool_size=4, strides=1)(x3)
x3 = Dropout(0.2)(x3)
x3 = GRU(128, return_sequences=True, activation='tanh')(x3)
x3= Dropout(0.2)(x3)
x3 = GRU(64, return_sequences=True, activation='tanh')(x3)
x3= Dropout(0.2)(x3)
x3 = GRU(32, return_sequences=True, activation='tanh')(x3)
x3 = Attention()(x3)
x3 = Activation(activation='softmax')(x3)

x123 = Multiply()([x12,x3])

embedded_input_s = Embedding(27,27,input_length=452)(main_input)
embedded_input_s = Dropout(0.2)(embedded_input_s)
x_s = GRU(128, return_sequences=True, activation='tanh')(embedded_input_s)
x_s = Dropout(0.2)(x_s)
x_s = GRU(32, return_sequences=True, activation='tanh')(x_s)
x_s = Attention()(x_s)

x = concatenate([x1,x12,x123,x_s])

x = Dense(256)(x)
x = Dropout(0.2)(x)
x = LeakyReLU()(x)

x = Dense(256)(x)
x = Dropout(0.2)(x)
x = LeakyReLU()(x)

x = Dense(128)(x)
x = Dropout(0.2)(x)
x = LeakyReLU()(x)

x = Dense(64)(x)
x = Dropout(0.2)(x)
x = LeakyReLU()(x)

output = Dense(12, activation='softmax')(x)
model = Model(inputs= main_input, outputs= output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

filepath="models/Hybrid_prunedCNN_3GRU_loss_{loss:.4f}-acc_{acc:.4f}-vloss_{val_loss:.4f}-vacc_{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

print model.summary()
plot_model(model, to_file='./plot/6_Hybrid_MultiHop.png', show_shapes=True, show_layer_names=False)

model.fit(xtra, ytra, epochs=100, validation_data=(xval,yval), batch_size=100, callbacks=callbacks_list)

yval_hat = model.predict(xval)
pr = calPrecisionRecall(yval, yval_hat)
