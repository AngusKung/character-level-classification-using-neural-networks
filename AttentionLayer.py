from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

# Below class is modified from:
# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py

# NOTE: Must use theano as Keras backend to use the class 
class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel',shape=(input_shape[-1],),initializer='normal',trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
