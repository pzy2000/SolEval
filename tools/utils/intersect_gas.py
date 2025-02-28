import json
import os
import pickle
from tqdm import tqdm
from pprint import pprint
result_folder = "results/gas"
jsonl_files = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.jsonl')]
pprint(jsonl_files)
GROUND_TRUTH_filename = '../../data/GROUND_TRUTH.jsonl'


with open(GROUND_TRUTH_filename, 'r') as f:
    ground_data = json.load(f)


intersect_list = []

for file_path, func_content in tqdm(ground_data.items(), colour='green'):
    patch_path = file_path.split('/')[-1]
    intersect_list.append(patch_path)


for json_file in jsonl_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    for file_path, func_content in tqdm(data.items(), colour='green'):
        patch_path = file_path.split('/')[-1]
        origin_patch_path = "patch/GROUND_TRUTH/" + patch_path
        if func_content["PASS"] == "False":
            # print("IF patch_path in intersect_list:", patch_path in intersect_list)
            if patch_path in intersect_list:
                intersect_list.remove(patch_path)
            # print("remove", patch_path)
            # print("if patch_path in intersect_list:", patch_path in intersect_list)

pickle.dump(intersect_list, open(os.path.join(result_folder, "intersect_gas.p"), "wb"))
