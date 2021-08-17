'''
@Author: your name
@Date: 2019-12-20 19:02:25
@LastEditTime: 2020-05-26 20:58:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /matengfei/KGCN_Keras-master/config.py
'''
# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'
#mdkg_hmdad
KG_FILE = {'mdkg_hmdad':os.path.join(RAW_DATA_DIR,'mdkg_hmdad','train2id.txt')}
ENTITY2ID_FILE = {'mdkg_hmdad':os.path.join(RAW_DATA_DIR,'mdkg_hmdad','entity2id.txt')}
EXAMPLE_FILE = {'mdkg_hmdad':os.path.join(RAW_DATA_DIR,'mdkg_hmdad','approved_example.txt')}

#add gaussian similarity
MICROBE_SIMILARITY_FILE = {'mdkg_hmdad':os.path.join(RAW_DATA_DIR,'mdkg_hmdad','microbesimilarity.txt')}
DISEASE_SIMILARITY_FILE = {'mdkg_hmdad':os.path.join(RAW_DATA_DIR,'mdkg_hmdad','diseasesimilarity.txt')}

SEPARATOR = {'mdkg_hmdad':'\t'}
NEIGHBOR_SIZE = {'mdkg_hmdad':8}


ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'

RESULT_LOG={'mdkg_hmdad':'mdkg_result.txt'}
PERFORMANCE_LOG = 'MDKGNN_performance.log'
DISEASE_MICROBE_EXAMPLE = 'dataset_examples.npy'



class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = None # neighbor sampling size
        self.embed_dim = None  # dimension of embedding
        self.n_depth = None    # depth of receptive field
        self.l2_weight = None # l2 regularizer weight
        self.lr = None  # learning rate
        self.batch_size = None
        self.aggregator_type = None
        self.n_epoch = None
        self.optimizer = None

        self.disease_vocab_size = None
        self.microbe_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None
        self.train_disease_similarity = None
        self.train_microbe_similarity = None
        self.test_disease_similarity = None
        self.test_microbe_similarity = None
        self.pre_embedding = None
        self.min_column = None


        self.exp_name = None
        self.model_name = None

        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'train_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'train_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='mdkg'
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
