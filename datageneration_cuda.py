
##################################
# library
##################################
import os
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

import cudf
import cugraph
from typing import List, Tuple, Dict, Optional, Set

os.environ["CUDA_VISIVLE_DEVICES"] = '2'

def to_cugraph(triples: List[Tuple[str, str, str]],
               entity_to_idx: Dict[str, int],
               predicate_to_idx: Dict[str, int],
               predicates: Optional[Set[str]] = None,
               is_multidigraph: bool = False) -> cugraph.Graph:
    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]

    # Extract nodes and edges
    src_nodes = [entity_to_idx[s] for s, p, o in _triples]
    dst_nodes = [entity_to_idx[o] for s, p, o in _triples]
    edges = cudf.DataFrame({'src': src_nodes, 'dst': dst_nodes})

    # Use cuGraph's MultiGraph if needed
    if is_multidigraph:
        G = cugraph.MultiGraph(directed=True)
        edges['p'] = [predicate_to_idx[p] for s, p, o in _triples]
        G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='p')
    else:
        # Remove duplicate edges for non-multigraph
        edges = edges.drop_duplicates().reset_index(drop=True)
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(edges, source='src', destination='dst')

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
    mG = to_cugraph(train, ent2id, rel2id, None, is_multidigraph=True)

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
    
    # Get the out-degree for all nodes in the graph
    degree_df = mG.out_degree()

    # Iterate over each node in the entity list and get the out-degree
    for i in entity_list:
        # Filter the degree_df DataFrame to get the out-degree of the specific node
        num_successors = degree_df[degree_df['vertex'] == i]['degree'].iloc[0]
        successors.append(num_successors)
    


    ##################################
    # final dataset dictionary
    ##################################
    entity_dict_whole = {}
    relation_dict_whole = {}

    query_data = {} # 최종 query data
    answer_data = {} # 최종 answer data


    query_type = args.query
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

    # Retrieve the edges as a DataFrame
    edges_df = mG.view_edge_list()


    for q_type in query_type_list:
        
        # queries = []
        # answer = []
        # bounded_entities = []
        query_to_answers = {}
        entity_dict = {}
        relation_dict = {}
        entity_candidates = sample_candidate(entity_num, successors_weight)
        if q_type == "1p":
            print("1p query generating ...")

            for i in tqdm(entity_candidates):
                # 1p
                node = i

                # Filter out the edges for the specific node
                edges1 = edges_df[edges_df['src'] == node]
                # edges1 = mG.out_edges(node, keys=False, data=True)

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):

                    start = row.src
                    end = row.dst
                    rel1 = (row.p,)

                # for itr_count1, (start, end, data1) in enumerate(edges1):
                #     if diversity_entity:
                #         if itr_count1 >= wgh1:
                #             break
                #     rel1 = tuple(data1.values())

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
            print("2p query generating ...")

            for i in tqdm(entity_candidates):
                node = i

                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start = row.src
                    bounded1 = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['src'] == bounded1]
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        end = row.dst
                        rel2 = (row.p,)

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        # loop query 방지
                        if end == start:
                            continue
                        if rel2[0] % 2 == 0 and rel1[0] == rel2[0]+1:
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
            print("3p query generating ...")

            for i in tqdm(entity_candidates):

                node = i

                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start = row.src
                    bounded1 = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['src'] == bounded1]
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        bounded2 = row.dst
                        rel2 = (row.p,)

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        # loop query 방지
                        if bounded2 == start:
                            continue
                        if rel2[0] % 2 == 0 and rel1[0] == rel2[0]+1: # rel2가 짝수면 바로 다음 홀수랑 pair이므로 rel1이 rel2보다 바로 다음으로 더 큰 홀수인지 확인
                            continue
                        elif rel1[0] == rel2[0]-1:
                            continue

                        if diversity_rel:
                            if relation_dict_whole.get(rel2[0], 0) >= limit:
                                continue
                            if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                continue

                        # 세 번째 edge
                        edges3 = edges_df[edges_df['src'] == bounded2]
                        edges3_pandas = edges3.to_pandas()

                        if sort:
                            edges3 = edges3.sort_values(by='p', ascending=False)

                        for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                            end = row.dst
                            rel3 = (row.p,)

                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break


                            # loop query 방지
                            if end == start or end == bounded1:
                                continue
                            if rel3[0] % 2 == 0 and rel2[0] == rel3[0]+1:
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
            print("2i query generating ...")
            for i in tqdm(entity_candidates):
                node = i
                
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break
                    
                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

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
            print("3i query generating ...")
            for i in tqdm(entity_candidates):
            
                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue

                    num_rel = 0
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 in-edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        if start2 == start1:
                            continue

                        if num_rel == 0:
                            start2_1 = start2
                            rel2 = (row.p,) # 두 번째 relation

                            if diversity_rel:
                                if relation_dict_whole.get(rel2[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                    continue
                        elif num_rel == 1:
                            start2_2 = start2
                            rel3 = (row.p,) # 세 번째 relation

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
            print("ip query generating ...")
            for i in tqdm(entity_candidates):

                prepare = {}
                node = i

                '''
                2i 생성 후, p 연결
                '''

                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

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
                    edges3 = edges_df[edges_df['src'] == bounded1]
                    edges3_pandas = edges3.to_pandas()

                    if sort:
                        edges3 = edges3.sort_values(by='p', ascending=False)

                    for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                        end = row.dst
                        rel3 = (row.p,)

                        if diversity_entity:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

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
            print("2in query generating ...")
            for i in tqdm(entity_candidates):
                node = i

                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

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
            print("3in query generating ...")
            for i in tqdm(entity_candidates):
            
                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    num_rel = 0
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 in-edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break

                        if start2 == start1:
                            continue

                        if num_rel == 0:
                            start2_1 = start2
                            rel2 = (row.p,) # 두 번째 relation
                            if diversity_rel:
                                if relation_dict_whole.get(rel2[0], 0) >= limit:
                                    continue
                                if relation_dict.get(rel2[0], 0) >= degree_wgh:
                                    continue
                        elif num_rel == 1:
                            start2_2 = start2
                            rel3 = (row.p,) # 세 번째 relation
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
            print("pi query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start = row.src
                    bounded1 = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['src'] == bounded1]
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        end = row.dst
                        rel2 = (row.p,)

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2[0] % 2 == 0 and rel1[0] == rel2[0]+1:
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

                        # 세 번째 edge
                        edges3 = edges_df[edges_df['dst'] == end]
                        edges3_pandas = edges3.to_pandas()

                        if sort:
                            edges3 = edges3.sort_values(by='p', ascending=False)

                        for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                            start2 = row.src
                            rel3 = (row.p,)

                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue


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
            print("inp query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}
                node = i

                '''
                2i 생성 후, p 연결
                '''
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

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
                    edges3 = edges_df[edges_df['src'] == bounded1]
                    edges3_pandas = edges3.to_pandas()

                    if sort:
                        edges3 = edges3.sort_values(by='p', ascending=False)

                    for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                        end = row.dst
                        rel3 = (row.p,)

                        if diversity_entity:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

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
            print("pin query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start = row.src
                    bounded1 = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['src'] == bounded1]
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        end = row.dst
                        rel2 = (row.p,)

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2[0] % 2 == 0 and rel1[0] == rel2[0]+1:
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

                        # 세 번째 edge
                        edges3 = edges_df[edges_df['dst'] == end]
                        edges3_pandas = edges3.to_pandas()

                        if sort:
                            edges3 = edges3.sort_values(by='p', ascending=False)

                        for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                            start2 = row.src
                            rel3 = (row.p,)

                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue
                            

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
            print("pni query generating ...")
            for i in tqdm(entity_candidates):
                prepare = {}

                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start = row.src
                    bounded1 = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue

                    # 두 번째 edge
                    edges2 = edges_df[edges_df['src'] == bounded1]
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        end = row.dst
                        rel2 = (row.p,)

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break

                        # loop query 방지
                        if end == start1:
                            continue
                        if rel2[0] % 2 == 0 and rel1[0] == rel2[0]+1:
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

                        # 세 번째 edge
                        edges3 = edges_df[edges_df['dst'] == end]
                        edges3_pandas = edges3.to_pandas()

                        if sort:
                            edges3 = edges3.sort_values(by='p', ascending=False)

                        for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                            start2 = row.src
                            rel3 = (row.p,)
                            
                            if diversity_entity:
                                if itr_count3 >= wgh2:
                                    break
                            if start2 == q_2p[0] or start2 in set(prepare[list_2p[0]]['bounded']):
                                continue
                            

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
            print("2u query generating ...")

            for i in tqdm(entity_candidates):
                node = i
                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh1:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh1:
                                break
                        if start2 == start1:
                            continue

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

                # 첫 번째 edge
                edges1 = edges_df[edges_df['src'] == node]
                # Convert the cuDF DataFrame to a pandas DataFrame
                edges1_pandas = edges1.to_pandas()

                if sort:
                    edges1 = edges1.sort_values(by='p', ascending=False)

                # Iterate using itertuples
                for itr_count1, row in enumerate(edges1_pandas.itertuples(index=False)):
                    start1 = row.src
                    end = row.dst
                    rel1 = (row.p,) # 첫 번째 relation

                    if diversity_entity:
                        if itr_count1 >= wgh2:
                            break

                    if diversity_rel:
                        if relation_dict_whole.get(rel1[0], 0) >= limit:
                            continue
                        if relation_dict.get(rel1[0], 0) >= degree_wgh:
                            continue
                    
                    # 두 번째 edge
                    edges2 = edges_df[edges_df['dst'] == end] # i type이므로 end의 edge를 구한다.

                    # Convert the cuDF DataFrame to a pandas DataFrame
                    edges2_pandas = edges2.to_pandas()

                    if sort:
                        edges2 = edges2.sort_values(by='p', ascending=False)

                    # Iterate using itertuples
                    for itr_count2, row in enumerate(edges2_pandas.itertuples(index=False)):
                        start2 = row.src
                        rel2 = (row.p,) # 두 번째 relation

                        if diversity_entity:
                            if itr_count2 >= wgh2:
                                break
                        if start2 == start1:
                            continue

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
                    edges3 = edges_df[edges_df['src'] == bounded1]
                    edges3_pandas = edges3.to_pandas()

                    if sort:
                        edges3 = edges3.sort_values(by='p', ascending=False)

                    for itr_count3, row in enumerate(edges3_pandas.itertuples(index=False)):
                        end = row.dst
                        rel3 = (row.p,)

                        if diversity_rel:
                            if itr_count3 >= wgh2:
                                break
                        if end == start1 or end == start2:
                            continue

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

    for i in range(len(list(query_data.keys()))):
        print(query_structure_to_type[list(query_data.keys())[i]], len(query_data[list(query_data.keys())[i]]))

    print(len(entity_dict_whole.keys()))
    print(len(relation_dict_whole.keys()))

    with open(f'data/random_query_{args.data}.pickle', 'wb') as file:
        pickle.dump(query_data, file)
    with open(f'data/random_query_answer_{args.data}.pickle', "wb") as file1:
        pickle.dump(answer_data, file1)

    ##################################
    # 결과 확인
    ##################################
    # generated_query = pd.read_pickle(f'data/random_query_{args.data}.pickle')
    # print(generated_query)
    # generated_query_answer = pd.read_pickle(f'data/random_query_answer_{args.data}.pickle')
    # print(generated_query_answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data generating...")
    parser.add_argument('--data', type=str, default="dummy_0002")
    parser.add_argument('--query', type=str, default="1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up")
    args = parser.parse_args()

    main(args)

