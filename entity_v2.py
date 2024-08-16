'''
This is for calculation of the metric from the result
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# bertv

import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import ast


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def main():
    # output path
    # output_dir = "entity"
    # print("Save results to: ", output_dir)


    # # prediction path

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # force = False
    # prediction_file = os.path.join(
    #     output_dir, f"entity_results.jsonl"
    # )
    # f_result, processed_results = get_output_file(prediction_file, force=force)
    # last_point = len(processed_results)
    # print("start_point", last_point)

    ent2desc = pd.read_table("data/ent2descript.txt", header=None)
    with open("data/ent2txt.json", 'r') as f:
        ent2txt = json.loads(f.read())
    ent2desc[0] = ent2desc[0].apply(lambda x: ent2txt[x])

    print(len(ent2desc))


    pred_ans = []
    actual_ans = []


    # JSON Lines 파일 읽기
    file_path = "entity/entity_results.jsonl"

    with open(file_path, 'r') as fpred:
        data = [json.loads(line) for line in fpred]

    print(data)

    exit(1)


    data = pd.read_parquet("data/rog_original_0007.parquet")
    # data = data.iloc[500000:, :].reset_index(drop=True)
    print(data)

    print(len(data))

    for i in range(last_point, len(data)):

        pred_ans.append((data['query_type'][i],data['query'][i]))
        actual_ans.append((data['query_type'][i],data['q_entity'][i]))


    print("Dataset: ", len(pred_ans))

    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    enti_emb = []

    for i in tqdm(range(len(ent2desc))):
        sentences = ent2desc[1][i]
        embeddings = embedder.encode([ent2desc[0][i] + sentences])[0] # 단어와 단어에 대한 description까지 사용해서 embedding 생성

        enti_emb.append(embeddings)

    enti_emb = np.array(enti_emb)  # 리스트를 numpy 배열로 변환


    pred_ans = pd.DataFrame(pred_ans)
    actual_ans = pd.DataFrame(actual_ans)

    FOLinExtracted = []
    FOLsameExtracted = []
    NOTin = []


    # 전체 임베딩을 GPU로 이동
    enti_emb = torch.tensor(enti_emb).cuda()
    enti_emb = F.normalize(enti_emb, p=2, dim=1)

    for i in tqdm(range(len(pred_ans))):
        k = len(actual_ans[1][i]) # 실제 head entity 개수
        # 비교할 sample의 임베딩 생성
        sample_sentence = pred_ans[1][i]
        sample_embedding = embedder.encode([sample_sentence], convert_to_tensor=True).cuda()

        # Normalize the embeddings to get cosine similarity
        sample_embedding = F.normalize(sample_embedding, p=2, dim=1)
        # enti_emb = F.normalize(enti_emb, p=2, dim=1)

        # 코사인 유사도 계산
        similarities = torch.matmul(sample_embedding, enti_emb.T)[0]

        # # 코사인 유사도 계산
        # similarities = cosine_similarity(sample_embedding.cpu().numpy(), enti_emb.cpu().numpy())[0]
        # 유사도를 기준으로 순위 매기기
        ranking = torch.argsort(similarities, descending=True)  # 내림차순 정렬
        # ranking = np.argsort(similarities)[::-1]  # 내림차순 정렬

        # top k 개 entity의 index 추출
        top_index = ranking[:k]
        
        # similarity value
        top_similarity = similarities[top_index]
        
        # corresponding entity
        top_entity = [','.join([ent2desc[0][index.item()] for index in top_index])]
        # top_entity = [','.join(ent2desc[0][top_index])]

        # print(f"Top Index: {top_index}, Top Entity: {top_entity}, Answer Entity: {actual_ans[1][i]} Top Similarity: {top_similarity}")

        rank = []
        for j in range(k):
            answer = actual_ans[1][i][j]
            # print("answer: ", answer)
            try:
                # matching_indices = torch.nonzero(ent2desc[0] == answer, as_tuple=False)
                idx = ent2desc[ent2desc[0] == answer].index.tolist()
                # print("matching index: ", idx)

                if len(idx) > 0:
                    # idx = matching_indices[0].item()
                    rank_position = torch.where(ranking == idx[0])[0].item()
                    # print(rank_position)
                    rank.append(rank_position)
                else:
                    rank.append(-1)
            except Exception as e:
                rank.append(-1)
                print(f"Exception occurred: {e}")


        # rank = []
        # for j in range(k):
        #     answer = actual_ans[1][i][j]
        #     try:
        #         idx = ent2desc[ent2desc[0] == answer].index[0]
        #         rank.append(np.where(ranking == idx)[0][0])
        #     except:
        #         rank.append(-1)
        #         pass
        
        all_smaller = all(x < k for x in rank)
        any_smaller = any(x < k for x in rank)

        data = {
            "id": i,
            "q_type": actual_ans[0][i],
            "nl": sample_sentence,
            "answer": list(actual_ans[1][i]),
            "prediction": top_entity,
            "similarity": [float(value) for value in top_similarity],
            # "ranking": [int(value) for value in rank],
            "ranking": rank,
            "FOLinExtracted": any_smaller+0,
            "FOLsameExtracted": all_smaller+0, 
            "ExtractedinFOL": 1-any_smaller,
        }

        f_result.write(json.dumps(data) + "\n")
        f_result.flush()
        
        FOLinExtracted.append((pred_ans[0][i],any_smaller+0))
        FOLsameExtracted.append((pred_ans[0][i], all_smaller+0))
        NOTin.append((pred_ans[0][i], 1-all_smaller))

        if i % 5000 == 0:
            print("Step: ", i)

            dataframes = [
                ("FOLinExtracted", pd.DataFrame(FOLinExtracted).groupby(0)[1].mean()),
                ("FOLsameExtracted", pd.DataFrame(FOLsameExtracted).groupby(0)[1].mean()),
                ("NOTin", pd.DataFrame(NOTin).groupby(0)[1].mean())
            ]
            
            for df_name, df in dataframes:
                ap = []
                an = []
                for query in df.index:
                    print(query, ": ", df[query])
                    if query in ['2in', '3in', 'inp', 'pin', 'pni']:
                        an.append(df[query])
                    else:
                        ap.append(df[query])
                if ap:
                    print(f"Ap {df_name}: {np.mean(ap)}")
                if an:
                    print(f"An {df_name}: {np.mean(an)}")

    print("##################### Final #######################")
    dataframes = [
        ("FOLinExtracted", pd.DataFrame(FOLinExtracted).groupby(0)[1].mean()),
        ("FOLsameExtracted", pd.DataFrame(FOLsameExtracted).groupby(0)[1].mean()),
        ("NOTin", pd.DataFrame(NOTin).groupby(0)[1].mean())
    ]
    
    for df_name, df in dataframes:
        ap = []
        an = []
        for query in df.index:
            print(query, ": ", df[query])
            if query in ['2in', '3in', 'inp', 'pin', 'pni']:
                an.append(df[query])
            else:
                ap.append(df[query])
        if ap:
            print(f"Ap {df_name}: {np.mean(ap)}")
        if an:
            print(f"An {df_name}: {np.mean(an)}")
    print("#####################################################")

    f_result.close()

if __name__ == '__main__':
    main()