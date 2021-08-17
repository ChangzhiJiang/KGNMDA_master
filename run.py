# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
from collections import defaultdict
sys.path.append(os.getcwd()) #add the env path
from sklearn.model_selection import train_test_split,StratifiedKFold
from main import train
from config import DISEASE_MICROBE_EXAMPLE, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ENTITY2ID_FILE, KG_FILE, \
    EXAMPLE_FILE, ENTITY_VOCAB_TEMPLATE, RESULT_LOG, MICROBE_SIMILARITY_FILE, DISEASE_SIMILARITY_FILE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, TRAIN_DATA_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from utils import pickle_dump, format_filename,write_log,pickle_load

def read_entity2id_file(file_path: str, entity_vocab: dict):
    print(f'Logging Info - Reading entity2id file: {file_path}' )
    assert len(entity_vocab) == 0
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if(count==0):
                count+=1
                continue
            _, entity = line.strip().split('\t')
            entity_vocab[entity] = len(entity_vocab)#entity_vocab:{'0':0,...}

def read_example_file(file_path:str,separator:str,entity_vocab:dict):
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(entity_vocab)>0
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            d1,d2,flag=line.strip().split(separator)[:3]
            if d1 not in entity_vocab or d2 not in entity_vocab:
                continue
            if d1 in entity_vocab and d2 in entity_vocab:
                examples.append([entity_vocab[d1],entity_vocab[d2],int(flag)])
    
    examples_matrix=np.array(examples)
    print(f'size of example: {examples_matrix.shape}')

    return examples_matrix

def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if count==0:
                count+=1
                continue
           # head, tail, relation = line.strip().split(' ') 
            head, relation, tail = line.strip().split('\t')
            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    random.seed(1)
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        if n_neighbor > 0:
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )

            adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])
    print('adj_entity=',adj_entity)
    return adj_entity, adj_relation
#gaussian similarity
def gaussian_similarity(interaction):
    inter_matrix = interaction
    nd = len(interaction)
    nm = len(interaction[0])
    #generate disease similarity
    kd = np.zeros((nd, nd))
    gama_d = nd / (pow(np.linalg.norm(inter_matrix),2))
    d_matrix = np.dot(inter_matrix,inter_matrix.T)
    for i in range(nd):
        j = i
        while j < nd:
            kd[i,j] = np.exp(-gama_d * (d_matrix[i,i] + d_matrix[j,j] - 2*d_matrix[i,j]))
            j += 1
    kd = kd + kd.T - np.diag(np.diag(kd))
    #generate microbe similarity
    km = np.zeros((nm,nm))
    gama_m = nm / (pow(np.linalg.norm(inter_matrix),2))
    m_matrix = np.dot(inter_matrix.T, inter_matrix)
    for l in range(nm):
        k = l
        while k < nm:
            km[l,k] = np.exp(-gama_m * (m_matrix[l,l] + m_matrix[k,k] - 2*m_matrix[l,k]))
            k += 1
    km = km + km.T - np.diag(np.diag(km))
    #print('kd=',kd,'km=',km)
    return kd,km
def generate_interaction(pairs_array):
    # first column:disease, second column:microbe, third column:0 or 1
    first_term2id, second_term2id = generate_dict_id(pairs_array)
    # interaction = np.zeros((disease_num, microbe_num))
    print('len(first_term2id)=',len(first_term2id),'len(second_term2id)=',len(second_term2id))
    print('first_term2id=',first_term2id,'second_term2id=',second_term2id)
    interaction = np.zeros((len(first_term2id), len(second_term2id)))
    for i in range(len(pairs_array)):
        if pairs_array[i,2] == 1:
            interaction[first_term2id[pairs_array[i,0]],second_term2id[pairs_array[i,1]]] = 1
    return interaction,first_term2id, second_term2id
#generate dict id for disease and microbe
def generate_dict_id(approved_data):
    #approved_data[:,:1]:disease, approved_data[:,1:2]:microbe, approved_data[:,2:3]:label
    first_term = set()
    second_term = set()
    for i in range(len(approved_data)):
        if approved_data[i,2] == 1:
            first_term.add(approved_data[i, 0])
            second_term.add(approved_data[i, 1])
    first_term2id = {}
    first_id = 0
    # first_termid = open('disease2id.txt','w')
    for term in first_term:
        first_term2id[term] = first_id
        # first_termid.write(str(term)+'\t'+str(first_id)+'\n')
        first_id += 1
    # first_termid.close()
    second_term2id = {}
    second_id = 0
    # second_termid = open('microbe2id.txt','w')
    for term in second_term:
        second_term2id[term] = second_id
        # second_termid.write(str(term)+'\t'+str(second_id)+'\n')
        second_id += 1
    # second_termid.close()
    return first_term2id, second_term2id

def generate_gaussian_file(all_data,test_data,disease_similarity_file,microbe_similarity_file):
    interaction, disease_term2id, microbe_term2id = generate_interaction(np.array(all_data))
    for i in range(len(test_data)):
        if test_data[i,2] == 1:
            interaction[disease_term2id[test_data[i,0]],microbe_term2id[test_data[i,1]]] = 0
    gaussian_d,gaussian_m = gaussian_similarity(interaction)
    disease_id2term = {value:key for key,value in disease_term2id.items()}
    microbe_id2term = {value:key for key,value in microbe_term2id.items()}
    disease_similarity = open(disease_similarity_file,'w')
    for i in range(len(gaussian_d)):
        disease_similarity.write(str(disease_id2term[i])+':')
        for j in range(len(gaussian_d[i])):
            if j != len(gaussian_d[i])-1:
                disease_similarity.write(str(gaussian_d[i][j])+'\t')
            if j == len(gaussian_d[i])-1:
                disease_similarity.write(str(gaussian_d[i][j])+'\n')
    disease_similarity.close()
    microbe_similarity = open(microbe_similarity_file,'w')
    for i in range(len(gaussian_m)):
        microbe_similarity.write(str(microbe_id2term[i])+':')
        for j in range(len(gaussian_m[i])):
            if j != len(gaussian_m[i])-1:
                microbe_similarity.write(str(gaussian_m[i][j])+'\t')
            if j == len(gaussian_m[i])-1:
                microbe_similarity.write(str(gaussian_m[i][j])+'\n')
    microbe_similarity.close()
    return 0

def process_data(dataset: str, neighbor_sample_size: int,K:int):

    entity_vocab = {}
    relation_vocab = {}

    read_entity2id_file(ENTITY2ID_FILE[dataset], entity_vocab)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),entity_vocab)#save entity_vocab

    examples_file=format_filename(PROCESSED_DATA_DIR, DISEASE_MICROBE_EXAMPLE, dataset=dataset)
    examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset],entity_vocab)
    np.save(examples_file,examples)#save examples
    
    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)#save entity_vocab
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)#save relation_vocab
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)#save adj_entity
    print('Logging Info - Saved:', adj_entity_file)
    
    disease_similarity_file = DISEASE_SIMILARITY_FILE[dataset]
    microbe_similarity_file = MICROBE_SIMILARITY_FILE[dataset]
    # test_disease_similarity_file = TEST_DISEASE_SIMILARITY_FILE[dataset]
    # test_microbe_similarity_file = TEST_MICROBE_SIMILARITY_FILE[dataset]
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)#save adj_relation
    print('Logging Info - Saved:', adj_entity_file)
    number_train = 10
    cv_total_auc = 0
    cv_total_aupr = 0
    cvs2_total_auc = 0
    cvs2_total_aupr = 0
    cvs3_total_auc = 0
    cvs3_total_aupr = 0
    for i in range(number_train):
        cv_auc, cv_aupr= cross_validation(K,examples,dataset,neighbor_sample_size, \
                                          disease_similarity_file,microbe_similarity_file)

        cv_total_auc += cv_auc
        cv_total_aupr += cv_aupr

    cv_average_auc = cv_total_auc / number_train
    cv_average_aupr = cv_total_aupr / number_train

    print(f'This is {K}_fold cv')
    print('cv_average_auc=',cv_average_auc,'cv_average_aupr=',cv_average_aupr)
    return 0


def cross_validation(K_fold,examples,dataset,neighbor_sample_size,disease_similarity_file,microbe_similarity_file):#self.K_Fold=1 do cross-validation
    subsets=dict()
    n_subsets=int(len(examples)/K_fold)
    remain=set(range(0,len(examples)))#examples:drug_vocab[d1] drug_vocab[d2] int(flag)(0 or 1)
    for i in reversed(range(0,K_fold-1)):
        subsets[i]=random.sample(remain,n_subsets)
        remain=remain.difference(subsets[i])
    subsets[K_fold-1]=remain
    #aggregator_types = ['sum_concat']
    aggregator_types = ['concat']
    print('aggregator_types=concat')
    #aggregator_types=['sum','concat','neigh']
    for t in aggregator_types:
        count=1
        temp={'dataset':dataset,'aggregator_type':t,'avg_auc':0.0,'avg_acc':0.0,'avg_f1':0.0,'avg_aupr':0.0}
        for i in reversed(range(0,K_fold)):
            test_data=examples[list(subsets[i])]
            #val_d,test_data=train_test_split(test_d,test_size=0.5)
            train_d=[]
            for j in range(0,K_fold):
                if i!=j:
                    train_d.extend(examples[list(subsets[j])])
            train_data=np.array(train_d)
            #generate_gaussian_file(train_data,disease_similarity_file,microbe_similarity_file)
            generate_gaussian_file(examples,test_data,disease_similarity_file,microbe_similarity_file)
            print('This is cross-validation S1.')
            print('len(train_data=)',len(train_data))
            print('len(test_data=)',len(test_data))
            train_log=train(
            kfold=count,
            dataset=dataset,
            train_d=train_data,
            test_d=test_data,
            neighbor_sample_size=neighbor_sample_size,
            embed_dim=32,
            n_depth=2,
            #n_depth=4,
            #l2_weight=5e-3,
            l2_weight=1e-1,
            lr=1e-3,
            #lr=1e-1,
            optimizer_type='adam',
            batch_size=32,
            aggregator_type=t,
            n_epoch=50,
            callbacks_to_add=['modelcheckpoint', 'earlystopping']
            )#train
            count+=1
            temp['avg_auc']=temp['avg_auc']+train_log['test_auc']
            temp['avg_acc']=temp['avg_acc']+train_log['test_acc']
            temp['avg_f1']=temp['avg_f1']+train_log['test_f1']
            temp['avg_aupr']=temp['avg_aupr']+train_log['test_aupr']
        for key in temp:
            if key=='aggregator_type' or key=='dataset':
                continue
            temp[key]=temp[key]/K_fold
        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]),temp,'a')
        print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')
    return temp['avg_auc'], temp['avg_aupr']

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    process_data('mdkg_hmdad',NEIGHBOR_SIZE['mdkg_hmdad'],5)



