# The attention layer must use theano backend
import os
os.environ["KERAS_BACKEND"]="theano"

import numpy as np
import pdb

from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from AttentionLayer import Attention
import utils

#SINGLE_ATTENTION_VECTOR=True

xtra, ytra, xval, yval = utils.readMemmap()

main_input = Input(shape= (xtra.shape[1],), dtype='float32')
masked_input = Masking(mask_value=0)(main_input)

embedded_input = Embedding(27,27,input_length=452)(masked_input)
embedded_input = Dropout(0.2)(embedded_input)
x1 = GRU(128, return_sequences=True, activation='tanh')(embedded_input)
x1 = Dropout(0.2)(x1)
x1 = GRU(64, return_sequences=True, activation='tanh')(x1)
x1 = Attention(x1,input_dim,time_steps,SINGLE_ATTENTION_VECTOR)
x1 = Attention()(x1)

embedded_input_s = Embedding(27,27,input_length=452)(masked_input)
embedded_input_s = Dropout(0.2)(embedded_input_s)
x_s = GRU(64, return_sequences=True, activation='tanh')(embedded_input_s)
x_s = Attention()(x_s)

x = concatenate([x1,x_s])

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

filepath="models/GRU_Attention_SmartMax_loss_{loss:.4f}-acc_{acc:.4f}-vloss_{val_loss:.4f}-vacc_{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

print model.summary()
plot_model(model, to_file='plot/4_GRU_Atten_Mask.png', show_shapes=True, show_layer_names=False)

model.fit(xtra, ytra, epochs=100, validation_data=(xval,yval), batch_size=100, callbacks=callbacks_list)

yval_hat = model.predict(xval)
pr = calPrecisionRecall(yval, yval_hat)
