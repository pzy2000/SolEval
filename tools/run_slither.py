from __future__ import absolute_import, division, print_function
import argparse
import itertools
import json
import os
import random
import warnings
from datetime import datetime
from typing import Union, List
import numpy as np
from tqdm import tqdm
from test_slither import analyze_contract
from utils.logger import MyLogger
from utils.replacements import retrieve_id


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    with open('rubbish_bin/raw_data.json', 'r') as file:
        data = json.load(file)
    if not os.path.exists("../patch/"):
        os.makedirs("../patch/")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=str, default="y")
    parser.add_argument("--rag", type=str, default="true")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--model", type=str, default="CodeLlama-34B")
    parser.add_argument("--verifier", type=str, default="")
    args = parser.parse_args()
    verifier = args.verifier
    # context_or_not = input("Do you want to use context? (y/n/c): ")
    context_or_not = args.context
    if context_or_not == "y":
        context = "context_True_testcase_False"
    elif context_or_not == "n":
        context = "context_False_testcase_False"
    elif context_or_not == "c":
        context = "context_False_testcase_True"
    elif context_or_not == "h":
        context = "context_True_testcase_True"
    else:
        raise NotImplementedError("Invalid input for context_or_not!!!")
    rag_or_random = args.rag
    if rag_or_random == "true":
        rag_path = "rag"
    elif rag_or_random == "false":
        rag_path = "random"
    else:
        raise NotImplementedError("Invalid input for rag_or_random!!!")
    # args.model = f"{args.model}_{context}"
    log_file = f"log_{args.model}_shot_{args.shot}_{context}_{current_time}.txt"
    if not os.path.exists("logs_slither/"):
        os.makedirs("logs_slither/")
    if not os.path.exists(f"logs_slither/{rag_path}/"):
        os.makedirs(f"logs_slither/{rag_path}/")
    if not os.path.exists("results/"):
        os.makedirs("results/")
    if not os.path.exists(f"results/{rag_path}/"):
        os.makedirs(f"results/{rag_path}/")
    logger = MyLogger(f"logs_slither/{rag_path}/{log_file}")
    logger.info_blue(f"Current context: {context}")
    set_seed(args.seed)
    num_return_sequences = args.sample
    import pickle
    real_path_cargo = pickle.load(open("../prebuilt/real_path_cargo.pkl", "rb"))
    logger.info("filter Over!")
    task_id = 0
    result = {}
    with open(verifier, "r") as file:
        veri_data = json.load(file)
    for file_path, file_content in tqdm(data.items(), colour='green'):
        # if "src/utils/DateTimeLib.sol" not in file_path:
        #     continue
        if file_path not in real_path_cargo.keys():
            continue
        if not file_content:
            continue
        if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
                or "test" in file_path or "forge" in file_path:
            continue
        logger.info_white("file_path:\n" + file_path)
        for i in range(len(file_content)):
            for method in tqdm(file_content[i]['methods'], colour='blue'):
                # print("file_path", file_path)
                # print("real_path_cargo[file_path]", real_path_cargo[file_path])
                # exit()
                if not file_content[i]['methods']:
                    continue
                task_id += 1
                identifier = method['identifier']
                if not os.path.exists(real_path_cargo[file_path]):
                    continue
                flag = retrieve_id(identifier, data[real_path_cargo[file_path]][0])
                if not method['full_signature'].startswith("function"):
                    logger.warn("jumping file_path:\n" + file_path + f" for TYPE is {method['full_signature']}")
                    continue
                if "human_labeled_comment" not in method.keys():
                    continue
                comment = method['human_labeled_comment']
                if not comment or not flag:
                    logger.warn(
                        "jumping file_path:\n" + file_path + f" for {bool(comment)} comment and {bool(flag)} flag")
                    continue
                function_full_sig = method['full_signature'].strip() + ' {' + '\n'
                logger.info_white("now testing function:\n" + function_full_sig)
                start = int(method['start'])
                end = int(method['end'])
                if end - start + 1 < 5:
                    logger.warn("jumping file_path:\n" + file_path + " for function length is " + str(end - start + 1))
                    continue
                try:
                    with open(f"{file_path}", 'r') as f:
                        source = f.readlines()
                except Exception as e:
                    logger.error("Error: " + str(e))
                    continue
                jump_flag = False
                for idx in tqdm(range(num_return_sequences), colour='yellow'):
                    if f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}" not in veri_data.keys():
                        # logger.error(f"patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx} not in veri_data!!!")
                        continue
                    if veri_data[f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}"]["PASS"] == "False":
                        continue
                    try:
                        with open(
                                f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt",
                                'r') as f:
                            patch = f.readlines()
                        with open(
                                f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt",
                                'r') as f:
                            patch_st = f.read()

                    except Exception as e:
                        logger.error("Error: " + str(e))
                        continue
                    jump_flag = True
                    patch_length = len(patch)
                    source_p = "\n".join(source[:start - 1] + patch + source[end:])
                    with open(f"{file_path}", 'r') as f:
                        source_bk = f.read()
                    file_path_bk = file_path.replace(".sol", ".sol.bak")
                    with open(f"{file_path_bk}", 'w') as f:
                        f.write(source_bk)
                    with open(f"{file_path}", 'w') as f:
                        f.write(source_p)
                    analysis = analyze_contract(file_path)
                    if analysis.get("error"):
                        print(f"Analyze fail: {analysis['message']}")
                    else:
                        print("Analyze success!")
                        print(f"Use compiler: {analysis['compiler_version']}")
                    with open(f"{file_path}", 'w') as f:
                        f.write(source_bk)
                    if analysis.get("error"):
                        logger.error("jumping due to analysis error!!!")
                        continue
                    result[f"{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}"] = analysis
                    # print("captured_stdout", captured_stdout)
                if not jump_flag:
                    logger.error("jumping due to jump_flag is False!!!")
                    continue
                logger.info("-----------------------------")
                with open(f'logs_slither/ALL_slither_{rag_path}_{args.model}_shot_{args.shot}_{context}.json', 'w') as file:
                    json.dump(result, file)
