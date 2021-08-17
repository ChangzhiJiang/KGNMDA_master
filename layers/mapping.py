# -*- coding: utf-8 -*-

from keras.engine.topology import Layer
from keras import backend as K

class DiseaseMicrobeScore(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(DiseaseMicrobeScore, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        input_disease_embed_dim = input_shape[0][-1]
        input_microbe_embed_dim = input_shape[1][-1]
        self.embed_dim = 32
        #self.hidden_1 = 128
        #self.hidden_2 = 64
        #self.embed_dim_second = 64
        #self.w_1 = self.add_weight(name=self.name + '_w_1',
        #                           shape=(input_disease_embed_dim+input_microbe_embed_dim, self.hidden_1),
        #                           initializer=self.initializer,regularizer=self.regularizer)
        #self.b_1 = self.add_weight(name=self.name + '_b_1',
        #                           shape=(self.hidden_1,),initializer='zeros')
        #self.w_2 = self.add_weight(name=self.name + '_w_2',
        #                           shape=(self.hidden_1, self.hidden_2),
        #                           initializer=self.initializer,regularizer=self.regularizer)
        #self.b_2 = self.add_weight(name=self.name + '_b_2',
        #                           shape=(self.hidden_2,),initializer='zeros')
        self.w_d = self.add_weight(name=self.name + '_w_d',
                                    shape=(input_disease_embed_dim,  self.embed_dim),
                                    initializer=self.initializer,regularizer=self.regularizer)
        # #self.b_d = self.add_weight(name=self.name + '_b_d',shape=(self.embed_dim,),initializer='zeros')
        self.w_m = self.add_weight(name=self.name + '_w_m',
                                  shape=(input_microbe_embed_dim,  self.embed_dim),
                                  initializer=self.initializer, regularizer=self.regularizer)
        #self.w_d_m = self.add_weight(name=self.name + '_w_d_m',
        #                          shape=(input_disease_embed_dim + input_microbe_embed_dim, self.embed_dim),
        #                          initializer=self.initializer,regularizer=self.regularizer)
        self.b_md = self.add_weight(name=self.name + '_b_md',shape=(self.embed_dim,),initializer='zeros')
        super(DiseaseMicrobeScore, self).build(input_shape)

    def call(self, inputs, **kwargs):
        disease, microbe = inputs
        #hidden = self.activation((K.dot(disease, self.w_d) * K.dot(microbe, self.w_m)) + self.b_md)
        #score = K.dot(hidden,  self.w_second) + self.b_second
        #score = self.activation(((K.dot(disease, self.w_d) * K.dot(microbe, self.w_m)) + (K.dot(K.concatenate([disease, microbe]), self.w_d_m))) + self.b_md)
        #score = self.activation((K.dot(disease, self.w_d) * K.dot(microbe, self.w_m)) + self.b_md)
        #hidden_layer1 = self.activation(K.dot(K.concatenate([disease,microbe]), self.w_1) + self.b_1)
        #hidden_layer2 = K.dot(hidden_layer1, self.w_2) + self.b_2
        score = K.dot(disease, self.w_d) * K.dot(microbe, self.w_m) + self.b_md
        #score = K.dot(K.concatenate([disease,microbe]), self.w_d_m) + self.b_md
        return score

    def compute_output_shape(self, input_shape):
        disease_shape, microbe_shape = input_shape
        return (disease_shape[0], self.embed_dim)
