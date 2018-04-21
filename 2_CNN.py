import numpy as np
import pdb

from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

import utils


xtra, ytra, xval, yval = utils.readMemmap()

main_input = Input(shape= (xtra.shape[1],), dtype='float32')

embedded_input = Embedding(27,27,input_length=452)(main_input)
embedded_input = Dropout(0.2)(embedded_input)
x1 = Conv1D(1, 16, strides=1, activation='relu')(embedded_input)
x1 = Dropout(0.2)(x1)
x1 = Conv1D(1, 8, strides=1, activation='relu')(x1)
x1 = MaxPooling1D(pool_size=8, strides=4)(x1)
x1 = Dropout(0.2)(x1)
x1 = Conv1D(1, 8, strides=1, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
x1 = Conv1D(1, 4, strides=1, activation='relu')(x1)
x1 = MaxPooling1D(pool_size=4, strides=2)(x1)
x1 = Dropout(0.2)(x1)
x1 = Conv1D(1, 4, strides=1, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
x1 = Conv1D(1, 4, strides=1, activation='relu')(x1)
x1 = MaxPooling1D(pool_size=2, strides=1)(x1)
x1 = Flatten()(x1)

x2 = Conv1D(1, 16, strides=1, activation='relu')(embedded_input)
x2 = MaxPooling1D(pool_size=8, strides=4)(x2)
x2 = Dropout(0.2)(x2)
x2 = Conv1D(1, 8, strides=1, activation='relu')(x2)
x2 = MaxPooling1D(pool_size=4, strides=2)(x2)
x2 = Dropout(0.2)(x2)
x2 = Conv1D(1, 4, strides=1, activation='relu')(x2)
x2 = MaxPooling1D(pool_size=2, strides=1)(x2)
x2 = Flatten()(x2)

x = concatenate([x1,x2])

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

filepath="models/CNN_loss_{loss:.4f}-acc_{acc:.4f}-vloss_{val_loss:.4f}-vacc_{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

print model.summary()
#plot_model(model, to_file='plot/2_CNN.png', show_shapes=True, show_layer_names=False)

model.fit(xtra, ytra, epochs=100, validation_data=(xval,yval), batch_size=100, callbacks=callbacks_list)

yval_hat = model.predict(xval)
pr = calPrecisionRecall(yval, yval_hat)
