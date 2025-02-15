import argparse
import json
import os
import warnings
from datetime import datetime

from tqdm import tqdm

from utils.logger import MyLogger

if __name__ == '__main__':
    GROUND_TRUTH_filename = '../rubbish_bin/GROUND_TRUTH.jsonl'
    with open(GROUND_TRUTH_filename, 'r') as file:
        ground_truth = json.load(file)
    result_folder = "results/gas"
    jsonl_files = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.jsonl')]
    intersect_list = []
    for file_path, func_content in tqdm(ground_truth.items(), colour='green'):
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
    print("length of intersect_list", len(intersect_list))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=str, default="y")
    parser.add_argument("--rag", type=str, default="true")
    parser.add_argument("--shot", type=int, default=1)
    # parser.add_argument("--model", type=str, default="CodeLlama-34B")
    args = parser.parse_args()
    # context_or_not = input("Do you want to use context? (y/n/c): ")
    jsonl_files = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.jsonl')]
    for json_file in jsonl_files:
        print("calculating ", json_file)
        filename = json_file
        context = filename.split('_')[5] + '_' + filename.split('_')[6]
        rag_or_random = args.rag
        if rag_or_random == "true":
            rag_path = "rag"
        elif rag_or_random == "false":
            rag_path = "random"
        else:
            raise NotImplementedError("Invalid input for rag_or_random!!!")
        model_name = filename.split('/')[-1]
        model_name = model_name.split('_')[1]
        model_name = f"{model_name}_{context}"
        log_file = f"log_{model_name}_shot_{args.shot}_{context}_{current_time}.txt"
        logger = MyLogger(f"logs_gas/{rag_path}/{log_file}")
        # logger.info_blue(f"Current model: {model_name}")
        # logger.info_blue(f"Current context: {context}")
        # logger.info_blue(f"Current shot: {args.shot}")
        # logger.info_blue(f"Current mode: {rag_or_random}")
        if not os.path.exists("logs_slither/"):
            os.makedirs("logs_slither/")
        if not os.path.exists(f"logs_slither/{rag_path}/"):
            os.makedirs(f"logs_slither/{rag_path}/")
        if not os.path.exists("results/"):
            os.makedirs("results/")
        if not os.path.exists(f"results/{rag_path}/"):
            os.makedirs(f"results/{rag_path}/")

        func_number = 0
        total_gas = 0
        with open(filename, 'r') as file:
            data = json.load(file)
        for file_path, func_content in tqdm(data.items(), colour='green'):
            patch_path = file_path.split('/')[-1]
            origin_patch_path = "patch/GROUND_TRUTH/" + patch_path
            if func_content["PASS"] == "False":
                continue
            if patch_path not in intersect_list:
                continue
            for test_func in func_content['GAS'].keys():
                gas_0 = int(func_content['GAS'][test_func]['gas']) if func_content['GAS'][test_func]['gas'] != 'None' else None
                gas_1 = int(func_content['GAS'][test_func]['~']) if func_content['GAS'][test_func]['~'] != 'None' else None
                gas_2 = int(func_content['GAS'][test_func]['μ']) if func_content['GAS'][test_func]['μ'] != 'None' else None
                gas_00 = int(ground_truth[origin_patch_path]['GAS'][test_func]['gas']) if gas_0 else None
                gas_10 = int(ground_truth[origin_patch_path]['GAS'][test_func]['~']) if gas_1 else None
                gas_20 = int(ground_truth[origin_patch_path]['GAS'][test_func]['μ']) if gas_2 else None
                total_gas += ((gas_00 - gas_0 if gas_00 else 0) + (gas_10 - gas_1 if gas_10 else 0)
                              + (gas_20 - gas_2 if gas_20 else 0))
            func_number += 1
            # exit(0)
            # print("total_gas: ", total_gas)
            # print("func_number: ", func_number)
        print("model_name:", model_name, ", average_gas: ", total_gas / func_number)
