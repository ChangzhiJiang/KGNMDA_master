# -*- coding: utf-8 -*-

from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from layers import Aggregator
from layers import DiseaseMicrobeScore
from callbacks import KGCNMetric
from models.base_model import BaseModel

class KGCN(BaseModel):
    def __init__(self, config):
        super(KGCN, self).__init__(config)

    def build(self):
        input_microbe = Input(shape=(1,), name='input_microbe', dtype='int64')#input microbe
        input_disease = Input(shape=(1,), name='input_disease', dtype='int64')#input_disease
        microbe_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='microbe_embedding')(input_microbe)
        disease_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='disease_embedding')(input_disease)
        # microbe_embedding = Lambda(lambda x:self.get_term_microbe_embedding(x))(input_microbe)
        # disease_embedding = Lambda(lambda x:self.get_term_disease_embedding(x))(input_disease)
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')
        receptive_list_microbe = Lambda(lambda x:self.get_receptive_field(x),
                                        name='receptive_filed_for_microbe')(input_microbe)
        neigh_ent_list_microbe = receptive_list_microbe[:self.config.n_depth+1]
        neigh_rel_list_microbe = receptive_list_microbe[self.config.n_depth+1:]
        neigh_ent_embed_list_microbe = [entity_embedding(neigh_ent) for neigh_ent in neigh_ent_list_microbe]
        neigh_rel_embed_list_microbe = [relation_embedding(neigh_rel) for neigh_rel in neigh_rel_list_microbe]

        neighbor_embedding = Lambda(lambda x:self.get_neighbor_info(x[0],x[1],x[2]),
                                    name='neigh_embedding_microbe')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}_drug_one'
            )
            next_neigh_ent_embed_list_microbe = []
            # next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed_microbe = neighbor_embedding([microbe_embedding, neigh_rel_embed_list_microbe[hop],
                                                     neigh_ent_embed_list_microbe[hop+1]])
                next_entity_embed_microbe = aggregator([neigh_ent_embed_list_microbe[hop],neighbor_embed_microbe])
                next_neigh_ent_embed_list_microbe.append(next_entity_embed_microbe)
            neigh_ent_embed_list_microbe = next_neigh_ent_embed_list_microbe


        receptive_list_disease = Lambda(lambda x: self.get_receptive_field(x),
                                        name='receptive_filed_for_disease')(input_disease)
        neigh_ent_list_disease = receptive_list_disease[:self.config.n_depth+1]
        neigh_rel_list_disease = receptive_list_disease[self.config.n_depth+1:]

        neigh_ent_embed_list_disease = [entity_embedding(neigh_ent) for neigh_ent in neigh_ent_list_disease]
        neigh_rel_embed_list_disease = [relation_embedding(neigh_rel) for neigh_rel in neigh_rel_list_disease]


        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}'
            )
            next_neigh_ent_embed_list_disease = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed_disease = neighbor_embedding([disease_embedding, neigh_rel_embed_list_disease[hop],
                                                     neigh_ent_embed_list_disease[hop+1]])
                next_entity_embed_disease = aggregator([neigh_ent_embed_list_disease[hop],neighbor_embed_disease])
                next_neigh_ent_embed_list_disease.append(next_entity_embed_disease)
            neigh_ent_embed_list_disease = next_neigh_ent_embed_list_disease


        microbe_squeeze_embed = Lambda(lambda x:K.squeeze(x,axis=1))(neigh_ent_embed_list_microbe[0])
        disease_squeeze_embed = Lambda(lambda x:K.squeeze(x,axis=1))(neigh_ent_embed_list_disease[0])

        print('microbe_squeeze_embed=',microbe_squeeze_embed)
        print('disease_squeeze_embed=',disease_squeeze_embed)
        # add disease and microbe gaussian similarity information
        microbe_pre_embedding = Lambda(lambda x:self.get_term_microbe_embedding(x))(input_microbe)
        disease_pre_embedding = Lambda(lambda x:self.get_term_disease_embedding(x))(input_disease)

        microbe_squeeze_pre_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(microbe_pre_embedding)
        disease_squeeze_pre_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(disease_pre_embedding)
        # final_microbe_embedding = Lambda(lambda x:x[0] * x[1])([microbe_squeeze_embed,microbe_squeeze_pre_embedding])
        # final_disease_embedding = Lambda(lambda x:x[0] * x[1])([disease_squeeze_embed,disease_squeeze_pre_embedding])
        final_microbe_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([microbe_squeeze_embed,microbe_squeeze_pre_embedding])
        final_disease_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([disease_squeeze_embed,disease_squeeze_pre_embedding])
        print('final_disease_embedding=',final_disease_embedding)
        print ('final_microbe_embedding=',final_microbe_embedding)
        disease_microbe_combine = DiseaseMicrobeScore(activation='tanh',
                regularizer=l2(self.config.l2_weight),
                name='diseasemicrobe_score')([final_disease_embedding,final_microbe_embedding])#MLP operation
        print('disease_microbe_combine=',disease_microbe_combine)
        disease_microbe_score = Lambda(lambda x:K.sigmoid(K.sum(x,axis=1,keepdims=True))
                                       )(disease_microbe_combine)
        # # disease_microbe_score = Lambda(lambda x:K.sigmoid(K.sum(x[0] * x[1], axis=1,keepdims=True))
        # #                                )([final_disease_embedding,final_microbe_embedding])
        # # disease_microbe_score = Lambda(lambda x:K.sigmoid(K.sum(x[0] * x[1], axis=1,keepdims=True))
        # #                                )([disease_squeeze_pre_embedding,microbe_squeeze_pre_embedding])
        print('disease_microbe_score=',disease_microbe_score)
        # disease_microbe_score = Lambda(lambda x:K.sigmoid(K.sum(x[0] * x[1], axis=1, keepdims=True)))([final_disease_embedding,final_microbe_embedding])
        # disease_microbe_score = Lambda(lambda x:K.sigmoid(K.sum(x[0] * x[1], axis=1, keepdims=True)))([disease_squeeze_embed,microbe_squeeze_embed])
        model = Model([input_disease, input_microbe],disease_microbe_score)

        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])
        return model
    def get_term_microbe_embedding(self, term):
        '''
        Gain pre_embedding for microbe
        :param term: input_microbe [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_microbe_embed_matrix = K.variable(self.config.microbe_pre_feature,name='pre_term_microbe_embedding',dtype='float32')
        microbe_pre_embed = K.gather(pre_microbe_embed_matrix,K.cast(term,dtype='int64'))
        return microbe_pre_embed
    def get_term_disease_embedding(self, term):
        '''
        Gain pre_embedding for disease
        :param term: input_disease, shape [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_disease_embed_matrix = K.variable(self.config.disease_pre_feature,name='pre_term_disease_embedding',dtype='float32')
        disease_pre_embed = K.gather(pre_disease_embed_matrix,K.cast(term,dtype='int64'))
        return disease_pre_embed


    def get_receptive_field(self, entity):
        """Calculate receptive field for entity using adjacent matrix

        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        #print('type(adj_relation_matrix)=',type(adj_relation_matrix))
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = K.relu(K.sum(drug * rel, axis=-1, keepdims=True))

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed

    def add_metrics(self, x_train, y_train):
        self.callbacks.append(KGCNMetric(x_train, y_train,self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train):
        self.callbacks = []
        self.add_metrics(x_train,y_train)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch,
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()

        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        return auc, acc, f1, aupr
    def gain_ytrue_ypredict(self,x,y):
        y_pred = self.model.predict(x).flatten()
        return y.flatten(),y_pred