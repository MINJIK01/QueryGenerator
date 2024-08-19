
##################################
# library
##################################

import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
import json
from typing import Tuple, List, Dict, Set, Optional
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse

##################################
# Function
##################################

def to_networkx(triples: List[Tuple[str, str, str]],
                entity_to_idx: Dict[str, int],
                predicate_to_idx: Dict[str, int],
                predicates: Optional[Set[str]] = None,
                is_multidigraph: bool = False) -> nx.DiGraph:
    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]

    G = nx.MultiDiGraph() if is_multidigraph else nx.DiGraph()

    entities = sorted({s for (s, _, _) in triples} | {o for (_, _, o) in triples})
    G.add_nodes_from([entity_to_idx[e] for e in entities])

    if is_multidigraph:
        G.add_edges_from([(entity_to_idx[s], entity_to_idx[o], {'p': predicate_to_idx[p]}) for s, p, o in _triples])
    else:
        edge_lst = sorted({(entity_to_idx[s], entity_to_idx[o]) for s, p, o in _triples})
        G.add_edges_from(edge_lst)

    return G

def query_counter(q_type, query):
    print(f"The number of {q_type} queries: ", len(query.keys()))

def sample_candidate(number_of_samples, successors_weight):
    return random.choices(range(0, 14505) , weights = successors_weight, k = number_of_samples)

query_structure_to_type = {('e', ('r',)): '1p',
                           ('e', ('r', 'r')): '2p',
                           ('e', ('r', 'r', 'r')): '3p',
                           (('e', ('r',)), ('e', ('r',))): '2i',
                           (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                           ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                           (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                           (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                           (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                           ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                           (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                           (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                           (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                           ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                           # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                           # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                           }

def main(args):
    ##################################
    # dataset
    ##################################
    # 변환 용 dict
    id2ent = pd.read_pickle("data/FB15k-237-betae/id2ent.pkl")
    id2rel = pd.read_pickle("data/FB15k-237-betae/id2rel.pkl")
    ent2id = {v:k for k, v in id2ent.items()}
    rel2id = {v:k for k, v in id2rel.items()}

    with open("data/ent2txt.json", 'r') as f:
        ent2txt = json.loads(f.read())

    # Q2T Graph 데이터 셋
    with open("data/FB15k-237-betae/train.txt") as f_train:
        train = f_train.read()
    train = train.split('\n')
    train = list(map(lambda x: (id2ent[int(x.split('\t')[0])], id2rel[int(x.split('\t')[1])], id2ent[int(x.split('\t')[2])]), train))


    # reasoning dataset
    org_train = pd.read_csv("data/FB15k-237-train.csv")
    org_valid = pd.read_csv("data/FB15k-237-valid.csv")
    org_test = pd.read_csv("data/FB15k-237-test.csv")

    org_train = org_train.iloc[:, 20::2].reset_index()
    org_valid = org_valid.iloc[:, 28::3].reset_index()
    org_test = org_test.iloc[:, 28::3].reset_index()

    references = pd.concat([org_train, org_valid, org_test]).reset_index(drop=True)




    ##################################
    # Graph Construction
    ##################################

    # MultiDirectedGraph
    mG = to_networkx(train, ent2id, rel2id, None, is_multidigraph=True)

    ##################################
    # parameters
    ##################################

    # degree_wgh이 더 작을 수록 각 query별 answer의 다양성이 높아짐

    degree_wgh_ctrl = 1000
    # wgh1= degree_wgh_ctrl
    # wgh2 = degree_wgh_ctrl
    wgh1= degree_wgh_ctrl ** (1/2)
    wgh2 = degree_wgh_ctrl ** (1/3)

    # degree_wgh이 더 작을 수록 각 query별 answer의 다양성이 높아짐
    degree_wgh = 3560

    # entity list
    entity_list = list(id2ent.keys())

    successors = []
    for i in entity_list:
        successors.append(len(list(mG.successors(i))))


    ##################################
    # final dataset dictionary
    ##################################
    entity_dict_whole = {}
    relation_dict_whole = {}

    query_data = {} # 최종 query data
    answer_data = {} # 최종 answer data


    query_type = "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"
    # query_type = "1p"
    query_type_list = [q for q in query_type.split('.')]

    successors_weight = list(map(lambda x: 1/x, successors))

    # number_of_samples = 1036
    # whole_candidates = random.choices(range(0, 14505) , weights = successors_weight, k = number_of_samples*14)

    sort = False
    suffle = True
    diversity_rel = False
    diversity_entity = True
    entity_num = 14505
    limit = 3860

    for q_type in query_type_list:
        
        # queries = []
        # answer = []
        # bounded_entities = []
        query_to_answers = {}
        if q_type == "1p":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)
            print("1p query generating ...")

            for i in tqdm(entity_candidates):
                # 1p
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                # print(edges1)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)
                # if suffle:
                #     edges1 = list(edges1)
                #     random.shuffle(edges1)
                #     print(edges1)


                for itr_count1, (start, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    rel1 = tuple(data1.values())

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue

                    query = (start, rel1)
                    if str(query) in set(references['1p_origin'].dropna().reset_index(drop=True).tolist()):
                        continue


                    if query not in query_to_answers:
                        query_to_answers[query] = {'answer': [], 'bounded': []}

                    query_to_answers[query]['answer'].append(end)

                    # Statistics calculation
                    entity_dict[start] = entity_dict.get(start, 0) + 1
                    relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                    entity_dict_whole[start] = entity_dict_whole.get(start, 0) + 1
                    relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                    # print(f"출발 노드: {start}, 도착 노드: {end}, 데이터: {tuple(rel)}")

            print("1p query generating finished")
            existing_set = query_data.get(('e', ('r',)), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[('e', ('r',))] = new_set

            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[('e', ('r',))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "2p":
            entity_dict = {}
            relation_dict = {}
            
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)
            print("2p query generating ...")

            for i in tqdm(entity_candidates):
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                for itr_count1, (start, bounded1, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    rel1 = tuple(data1.values())  # 첫 번째 edge

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.out_edges(bounded1, keys=False, data=True)

                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)
                    
                    for itr_count2, (_, end, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if end == start:
                            continue
                        if rel2 % 2 == 0 and rel1[0] == rel2[0]+1:
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        # print(f"출발 노드: {start}, 중간 노드: {bounded}, 도착 노드: {end}, 데이터: {rel1 + rel2}")
                        query = (start, rel1 + rel2)

                        if str(query) in set(references['2p_origin'].dropna().reset_index(drop=True).tolist()):
                            continue
                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        query_to_answers[query]['bounded'].append(bounded1)

                        # Statistics calculation
                        entity_dict[start] = entity_dict.get(start, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start] = entity_dict_whole.get(start, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                        
            print("2p query generating finished")
            existing_set = query_data.get(('e', ('r', 'r')), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[('e', ('r', 'r'))] = new_set

            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[('e', ('r', 'r'))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "3p":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)
            print("3p query generating ...")

            for i in tqdm(entity_candidates):

                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                for itr_count1, (start, bounded1, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())  # 첫 번째 edge

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.out_edges(bounded1, keys=False, data=True)

                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)
                    for itr_count2, (_, bounded2, data2) in enumerate(tqdm(edges2)):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if bounded2 == start:
                            continue
                        if rel2 % 2 == 0 and rel1[0] == rel2[0]+1: # rel2가 짝수면 바로 다음 홀수랑 pair이므로 rel1이 rel2보다 바로 다음으로 더 큰 홀수인지 확인
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue
                        # 세 번째 edge
                        edges3 = mG.out_edges(bounded2, keys=False, data=True)
                        for itr_count3, (_, end, data3) in enumerate(edges3):
                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break

                            rel3 = tuple(data3.values())

                            # loop query 방지
                            if end == start or end == bounded1:
                                continue
                            if rel3 % 2 == 0 and rel2 == rel3+1:
                                continue
                            elif rel2 == rel3-1:
                                continue

                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue
                            # print(f"출발 노드: {start}, 중간 노드: {(bounded1, bounded2)}, 도착 노드: {end}, 데이터: {rel1 + rel2 + rel3}")

                            query = (start, rel1 + rel2 + rel3)

                            if str(query) in set(references['3p_origin'].dropna().reset_index(drop=True).tolist()):
                                continue

                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}
                            query_to_answers[query]['answer'].append(end)
                            query_to_answers[query]['bounded'].append((bounded1, bounded2))

                            # Statistics calculation
                            entity_dict[start] = entity_dict.get(start, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start] = entity_dict_whole.get(start, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1
            print("3p query generating finished")
            existing_set = query_data.get(('e', ('r', 'r', 'r')), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[('e', ('r', 'r', 'r'))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[('e', ('r', 'r', 'r'))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "2i":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("2i query generating ...")
            for i in tqdm(entity_candidates):
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    rel1 = tuple(data1.values())
                    
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

                        rel2 = tuple(data2.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2))

                        if str(query) in set(references['2i_origin'].dropna().reset_index(drop=True).tolist()):
                            continue
                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        
                        # 결과 출력 및 저장
                        # print(f"출발 노드1: {start1}, 출발 노드2: {start2}, 도착 노드: {end}, 데이터: {rel1 + rel2}")

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1

            print("2i query generating finished")
            # query_data[(('e', ('r',)), ('e', ('r',)))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})

            existing_set = query_data.get((('e', ('r',)), ('e', ('r',))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r',)), ('e', ('r',)))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            query_counter(q_type, query_to_answers)
        elif q_type == "3i":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("3i query generating ...")
            for i in tqdm(entity_candidates):
            
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)
                
                # if sort:
                #     edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    rel1 = tuple(data1.values())

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    num_rel = 0
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)

                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        if start2 == start1:
                            continue

                        if num_rel == 0:
                            start2_1 = start2
                            rel2 = tuple(data2.values())

                            if diversity_rel:
                                if relation_dict_whole.get(rel2[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                    continue
                        elif num_rel == 1:
                            start2_2 = start2
                            rel3 = tuple(data2.values())

                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue
                        num_rel += 1
                            
                        if num_rel == 2:
                            query = ((start1, rel1), (start2_1, rel2), (start2_2, rel3))

                            if str(query) in set(references['3i_origin'].dropna().reset_index(drop=True).tolist()):
                                continue

                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}
                            query_to_answers[query]['answer'].append(end)
                            num_rel = 0
                            
                            # 결과 출력 및 저장
                            # print(f"출발 노드1: {start1}, 출발 노드2: {start2_1}, 출발 노드3: {start2_2}, 도착 노드: {end}, 데이터: {rel1 + rel2 + rel3}")

                            # Statistics calculation
                            entity_dict[start1] = entity_dict.get(start1, 0) + 1
                            entity_dict[start2_1] = entity_dict.get(start2_1, 0) + 1
                            entity_dict[start2_2] = entity_dict.get(start2_2, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                            entity_dict_whole[start2_1] = entity_dict_whole.get(start2_1, 0) + 1
                            entity_dict_whole[start2_2] = entity_dict_whole.get(start2_2, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1

            print("3i query generating finished")
            existing_set = query_data.get((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "ip":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("ip query generating ...")
            for i in tqdm(entity_candidates):

                prepare = {}
                node = i

                '''
                2i 생성 후, p 연결
                '''

                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    
                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)

                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if start2 == start1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2))


                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)

                list_2i = list(prepare.keys())
                boundeds = [x['answer'][0] for x in prepare.values()]

                for k in range(len(list_2i)):
                    
                    bounded1 = boundeds[k]
                    start1 = list_2i[k][0][0]
                    start2 = list_2i[k][1][0]
                    rel1 = (list_2i[k][0][1][0],)
                    rel2 = (list_2i[k][1][1][0],)

                    # 세 번째 edge
                    edges3 = mG.out_edges(bounded1, keys=False, data=True)
                    for itr_count3, (_, end, data3) in enumerate(edges3):
                        if diversity_entity:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

                        rel3 = tuple(data3.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel3[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                continue
                        query = (((start1, rel1), (start2, rel2)), rel3)

                        if str(query) in set(references['ip_origin'].dropna().reset_index(drop=True).tolist()):
                            continue

                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        query_to_answers[query]['bounded'].append(bounded1)

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                        relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                        relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1


            print("ip query generating finished")
            existing_set = query_data.get(((('e', ('r',)), ('e', ('r',))), ('r',)), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[((('e', ('r',)), ('e', ('r',))), ('r',))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[((('e', ('r',)), ('e', ('r',))), ('r',))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "2in":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("2in query generating ...")
            for i in tqdm(entity_candidates):
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    rel1 = tuple(data1.values())
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

                        rel2 = tuple(data2.values())
                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2 + (-2,)))

                        if str(query) in set(references['2in_origin'].dropna().reset_index(drop=True).tolist()):
                            continue
                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        
                        # 결과 출력 및 저장
                        # print(f"출발 노드1: {start1}, 출발 노드2: {start2}, 도착 노드: {end}, 데이터: {rel1 + rel2}")

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1

            print("2in query generating finished")
            existing_set = query_data.get((('e', ('r',)), ('e', ('r', 'n'))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r',)), ('e', ('r', 'n')))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r',)), ('e', ('r', 'n')))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "3in":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("3in query generating ...")
            for i in tqdm(entity_candidates):
            
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    rel1 = tuple(data1.values())
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    num_rel = 0
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)
                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        if start2 == start1:
                            continue

                        if num_rel == 0:
                            start2_1 = start2
                            rel2 = tuple(data2.values())
                            if diversity_rel:
                                if relation_dict_whole.get(rel2[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                    continue
                        elif num_rel == 1:
                            start2_2 = start2
                            rel3 = tuple(data2.values())
                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue
                        num_rel += 1
                            
                        if num_rel == 2:
                            query = ((start1, rel1), (start2_1, rel2), (start2_2, rel3 + (-2,)))

                            if str(query) in set(references['3in_origin'].dropna().reset_index(drop=True).tolist()):
                                continue
                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}
                            query_to_answers[query]['answer'].append(end)
                            num_rel = 0
                            
                            # 결과 출력 및 저장
                            # print(f"출발 노드1: {start1}, 출발 노드2: {start2_1}, 출발 노드3: {start2_2}, 도착 노드: {end}, 데이터: {rel1 + rel2 + rel3}")

                            # Statistics calculation
                            entity_dict[start1] = entity_dict.get(start1, 0) + 1
                            entity_dict[start2_1] = entity_dict.get(start2_1, 0) + 1
                            entity_dict[start2_2] = entity_dict.get(start2_2, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                            entity_dict_whole[start2_2] = entity_dict_whole.get(start2_2, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1
                        
            print("3in query generating finished")
            existing_set = query_data.get((('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "pi":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("pi query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                for itr_count1, (start1, bounded1, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())  # 첫 번째 edge

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.out_edges(bounded1, keys=False, data=True)

                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)

                    for itr_count2, (_, end, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break


                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2 % 2 == 0 and rel1[0] == rel2[0]+1:
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        # print(f"출발 노드: {start}, 중간 노드: {bounded}, 도착 노드: {end}, 데이터: {rel1 + rel2}")
                        query = (start1, rel1 + rel2)
                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)
                        prepare[query]['bounded'].append(bounded1)

                list_2p = list(prepare.keys())
                ends = [x['answer'] for x in prepare.values()]

                for k, q_2p in enumerate(list_2p):
                    # if k >= wgh2:
                    #     break

                    # for end in ends[k]: # 쿼리 하나당 생성되는 모든 answer에 대해서 pi를 생성하려면 사용

                    
                    for j, end in enumerate(ends[k]):

                        start1 = q_2p[0]
                        rel1 = (q_2p[1][0],)
                        rel2 = (q_2p[1][1],)

                        edges3 = mG.in_edges(end, keys=False, data=True)

                        for itr_count3, (start2, _, data) in enumerate(edges3):
                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue
                            
                            rel3 = tuple(data1.values())

                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue
                            
                            query = (q_2p, (start2, rel3))
                            if str(query) in set(references['pi_origin'].dropna().reset_index(drop=True).tolist()):
                                continue

                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}

                            query_to_answers[query]['answer'].append(end)
                            query_to_answers[query]['bounded'].append(prepare[q_2p]['bounded'][j])

                            # Statistics calculation
                            entity_dict[start1] = entity_dict.get(start1, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1

                            entity_dict[start2] = entity_dict.get(start2, 0) + 1
                            relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1

            print("pi query generating finished")
            existing_set = query_data.get((('e', ('r', 'r')), ('e', ('r',))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r', 'r')), ('e', ('r',)))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r', 'r')), ('e', ('r',)))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "inp":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("inp query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}
                node = i

                '''
                2i 생성 후, p 연결
                '''
                edges1 = mG.out_edges(node, keys=False, data=True)
                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)

                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)
                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if start2 == start1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2))

                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)

                list_2i = list(prepare.keys())
                boundeds = [x['answer'][0] for x in prepare.values()]

                for k in range(len(list_2i)):
                    
                    bounded1 = boundeds[k]
                    start1 = list_2i[k][0][0]
                    start2 = list_2i[k][1][0]
                    rel1 = (list_2i[k][0][1][0],)
                    rel2 = (list_2i[k][1][1][0],)

                    # 세 번째 edge
                    edges3 = mG.out_edges(bounded1, keys=False, data=True)
                    for itr_count3, (_, end, data3) in enumerate(edges3):
                        if diversity_entity:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

                        rel3 = tuple(data3.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel3[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                continue

                        query = (((start1, rel1), (start2, rel2 + (-2,))), rel3)

                        if str(query) in set(references['inp_origin'].dropna().reset_index(drop=True).tolist()):
                            continue

                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        query_to_answers[query]['bounded'].append(bounded1)

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                        relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                        relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1

            print("inp query generating finished")
            existing_set = query_data.get(((('e', ('r',)), ('e', ('r', 'n'))), ('r',)), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "pin":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("pin query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)
                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                for itr_count1, (start1, bounded1, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())  # 첫 번째 edge

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.out_edges(bounded1, keys=False, data=True)
                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)

                    for itr_count2, (_, end, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2 % 2 == 0 and rel1[0] == rel2[0]+1:
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue
                        
                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        # print(f"출발 노드: {start}, 중간 노드: {bounded}, 도착 노드: {end}, 데이터: {rel1 + rel2}")
                        query = (start1, rel1 + rel2)
                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)
                        prepare[query]['bounded'].append(bounded1)


                list_2p = list(prepare.keys())
                ends = [x['answer'] for x in prepare.values()]

                for k, q_2p in enumerate(list_2p):
                    # if k >= wgh2:
                    #     break

                    # for end in ends[k]: # 쿼리 하나당 생성되는 모든 answer에 대해서 pi를 생성하려면 사용
                    for j, end in enumerate(ends[k]):

                        start1 = q_2p[0]
                        rel1 = (q_2p[1][0],)
                        rel2 = (q_2p[1][1],)

                        edges3 = mG.in_edges(end, keys=False, data=True)

                        for itr_count3, (start2, _, data) in enumerate(edges3):
                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue
                            
                            rel3 = tuple(data1.values())

                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue

                            query = (q_2p, (start2, rel3 + (-2,)))
                            
                            if str(query) in set(references['pin_origin'].dropna().reset_index(drop=True).tolist()):
                                continue

                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}

                            query_to_answers[query]['answer'].append(end)
                            query_to_answers[query]['bounded'].append(prepare[q_2p]['bounded'][j])

                            # Statistics calculation
                            entity_dict[start1] = entity_dict.get(start1, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                            entity_dict[start2] = entity_dict.get(start2, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1
            print("pin query generating finished")
            existing_set = query_data.get((('e', ('r', 'r')), ('e', ('r', 'n'))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r', 'r')), ('e', ('r', 'n')))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r', 'r')), ('e', ('r', 'n')))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "pni":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("pni query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)
                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                for itr_count1, (start1, bounded1, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())  # 첫 번째 edge

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue

                    # 두 번째 edge
                    edges2 = mG.out_edges(bounded1, keys=False, data=True)
                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)
                    for itr_count2, (_, end, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        rel2 = tuple(data2.values())

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2 % 2 == 0 and rel1[0] == rel2[0]+1:
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        # print(f"출발 노드: {start}, 중간 노드: {bounded}, 도착 노드: {end}, 데이터: {rel1 + rel2}")
                        query = (start1, rel1 + rel2 + (-2,))
                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)
                        prepare[query]['bounded'].append(bounded1)

                list_2p = list(prepare.keys())
                ends = [x['answer'] for x in prepare.values()]

                for k, q_2p in enumerate(list_2p):
                    # if k >= wgh2:
                    #     break

                    # for end in ends[k]: # 쿼리 하나당 생성되는 모든 answer에 대해서 pi를 생성하려면 사용
                    for j, end in enumerate(ends[k]):

                        start1 = q_2p[0]
                        rel1 = (q_2p[1][0],)
                        rel2 = (q_2p[1][1],)

                        edges1 = mG.in_edges(end, keys=False, data=True)

                        for itr_count3, (start2, _, data) in enumerate(edges1):
                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue
                            
                            rel3 = tuple(data1.values())

                            if diversity_rel:
                                if relation_dict_whole.get(rel3[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                    continue

                            query = (q_2p, (start2, rel3))

                            if str(query) in set(references['pni_origin'].dropna().reset_index(drop=True).tolist()):
                                continue

                            if query not in query_to_answers:
                                query_to_answers[query] = {'answer': [], 'bounded': []}

                            query_to_answers[query]['answer'].append(end)
                            query_to_answers[query]['bounded'].append(prepare[q_2p]['bounded'][j])

                            # Statistics calculation
                            entity_dict[start1] = entity_dict.get(start1, 0) + 1
                            relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                            relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                            entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                            relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                            relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
                            entity_dict[start2] = entity_dict.get(start2, 0) + 1
                            relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                            entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                            relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1
            print("pni query generating finished")
            existing_set = query_data.get((('e', ('r', 'r', 'n')), ('e', ('r',))), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "2u":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)

            print("2u query generating ...")
            for i in tqdm(entity_candidates):
                node = i
                edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)


                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    rel1 = tuple(data1.values())

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)
                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

                        rel2 = tuple(data2.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2), (-1,))

                        if str(query) in set(references['2u_origin'].dropna().reset_index(drop=True).tolist()):
                            continue
                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1
            print("2u query generating finished")
            existing_set = query_data.get((('e', ('r',)), ('e', ('r',)), ('u',)), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[(('e', ('r',)), ('e', ('r',)), ('u',))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[(('e', ('r',)), ('e', ('r',)), ('u',))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)
        elif q_type == "up":
            entity_dict = {}
            relation_dict = {}
            # entity_candidates = whole_candidates[num_q*number_of_samples:num_q*number_of_samples+number_of_samples-1]
            entity_candidates = sample_candidate(entity_num, successors_weight)
            
            print("up query generating ...")
            for i in tqdm(entity_candidates):

                prepare = {}
                node = i

                '''
                2i 생성 후, p 연결
                '''

                edges1 = mG.out_edges(node, keys=False, data=True)
                if sort:
                    edges1 = sorted(edges1, key=lambda x: x[2]['p'], reverse=True)

                # 첫 번째 edge
                for itr_count1, (start1, end, data1) in enumerate(edges1):
                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break
                    rel1 = tuple(data1.values())
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = mG.in_edges(end, keys=False, data=True)

                    if sort:
                        edges2 = sorted(edges2, key=lambda x: x[2]['p'], reverse=True)

                    for itr_count2, (start2, _, data2) in enumerate(edges2):
                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break
                        if start2 == start1:
                            continue

                        rel2 = tuple(data2.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        query = ((start1, rel1), (start2, rel2))

                        if str(query) in set(references['up_origin'].dropna().reset_index(drop=True).tolist()):
                            continue    

                        # Statistics calculation
                        entity_dict[start1] = entity_dict.get(start1, 0) + 1
                        entity_dict[start2] = entity_dict.get(start2, 0) + 1
                        relation_dict[rel1[0]] = relation_dict.get(rel1[0], 0) + 1
                        relation_dict[rel2[0]] = relation_dict.get(rel2[0], 0) + 1
                        entity_dict_whole[start1] = entity_dict_whole.get(start1, 0) + 1
                        entity_dict_whole[start2] = entity_dict_whole.get(start2, 0) + 1
                        relation_dict_whole[rel1[0]] = relation_dict_whole.get(rel1[0], 0) + 1
                        relation_dict_whole[rel2[0]] = relation_dict_whole.get(rel2[0], 0) + 1

                        if query not in prepare:
                            prepare[query] = {'answer': [], 'bounded': []}
                        prepare[query]['answer'].append(end)

                list_2i = list(prepare.keys())
                boundeds = [x['answer'][0] for x in prepare.values()]

                for k in range(len(list_2i)):
                    
                    bounded1 = boundeds[k]
                    start1 = list_2i[k][0][0]
                    start2 = list_2i[k][1][0]
                    rel1 = (list_2i[k][0][1][0],)
                    rel2 = (list_2i[k][1][1][0],)

                    # 세 번째 edge
                    edges3 = mG.out_edges(bounded1, keys=False, data=True)
                    for itr_count3, (_, end, data3) in enumerate(edges3):
                        if diversity_rel:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

                        rel3 = tuple(data3.values())

                        if diversity_rel:
                            if relation_dict_whole.get(rel3[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel3[0], 0) >= degree_wgh:
                                continue
                            
                        query = (((start1, rel1), (start2, rel2), (-1,)), rel3)
                        if query not in query_to_answers:
                            query_to_answers[query] = {'answer': [], 'bounded': []}
                        query_to_answers[query]['answer'].append(end)
                        query_to_answers[query]['bounded'].append(bounded1)


                        # Statistics calculation
                        relation_dict[rel3[0]] = relation_dict.get(rel3[0], 0) + 1
                        relation_dict_whole[rel3[0]] = relation_dict_whole.get(rel3[0], 0) + 1
            print("up query generating finished")
            existing_set = query_data.get(((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)), set())
            new_set = existing_set.union(query_to_answers.keys())
            query_data[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = new_set
            for key, value in query_to_answers.items():
                if key in answer_data:
                    answer_data[key].update(value['answer'])
                else:
                    answer_data[key] = set(value['answer'])
            # query_data[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = set(query_to_answers.keys())
            # answer_data.update({key: set(value['answer']) for key, value in query_to_answers.items()})
            query_counter(q_type, query_to_answers)

    for i in range(14):
        print(query_structure_to_type[list(query_data.keys())[i]], len(query_data[list(query_data.keys())[i]]))

    print(len(entity_dict_whole.keys()))
    print(len(relation_dict_whole.keys()))

    with open(f'data/random_query_{args.data}.pickle', 'wb') as file:
        pickle.dump(query_data, file)
    with open(f'data/random_query_answer_{args.data}.pickle', "wb") as file1:
        pickle.dump(answer_data, file1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data generating...")
    parser.add_argument('--data', type=str, default="dummy_0002")
    args = parser.parse_args()

    main(args)

