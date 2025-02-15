from __future__ import absolute_import, division, print_function
import argparse
import itertools
import json
import os
import random
import re
import subprocess
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Union, List
import numpy as np
import torch
from tqdm import tqdm
from extract_function_from_solidity_project import serialize
from utils.logger import MyLogger
from utils.replacements import cwd_dir_cargo, retrieve_id, generate_replaced_paths, single_replacements, replacements


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    with open('data/raw_data.json', 'r') as file:
        data = json.load(file)
    if not os.path.exists("patch/"):
        os.makedirs("patch/")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=str, default="y")
    parser.add_argument("--rag", type=str, default="true")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--model", type=str, default="CodeLlama_7b")
    args = parser.parse_args()
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
    filename = f'results/{rag_path}/results_{args.model}_shot_{args.shot}_{context}_{current_time}.jsonl'
    log_file = f"log_{args.model}_shot_{args.shot}_{context}_{current_time}.txt"
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    if not os.path.exists(f"logs/{rag_path}/"):
        os.makedirs(f"logs/{rag_path}/")
    if not os.path.exists("results/"):
        os.makedirs("results/")
    if not os.path.exists(f"results/{rag_path}/"):
        os.makedirs(f"results/{rag_path}/")
    logger = MyLogger(f"logs/{rag_path}/{log_file}")
    logger.info_blue(f"Current context: {context}")
    set_seed(args.seed)
    num_return_sequences = args.sample
    log_dict = {}
    real_path_cargo = {}
    for file_path, file_content in tqdm(data.items()):
        file_path = file_path.replace("/root/", "repository/")
        if "forge" in file_path:
            continue
        if not (file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
                or "test" in file_path or "forge" in file_path):
            continue
        real_file_path_list = generate_replaced_paths(file_path, single_replacements, single=True)
        if not real_file_path_list:
            # print("re generate replaced_paths......")
            real_file_path_list = generate_replaced_paths(file_path, replacements)
        if not real_file_path_list:
            # print("not found any replaced_paths......")
            pass
        else:
            for real_file_path in real_file_path_list:
                # print("real_file_path:", real_file_path)
                real_path_cargo[real_file_path] = file_path
    logger.info("filter Over!")
    number_total = 0
    number_pass = 0
    number_fail = 0
    number_compiled_total = 0
    number_compiled_fail = 0
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    task_compiled_correct = defaultdict(int)
    task_id = 0
    for file_path, file_content in tqdm(data.items(), colour='green'):
        file_path = file_path.replace("/root/", "repository/")
        if file_path not in real_path_cargo.keys():
            continue
        if not file_content:
            continue
        if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
                or "test" in file_path or "forge" in file_path:
            continue
        logger.info_white("file_path:\n" + file_path)
        for method in tqdm(file_content, colour='blue'):
            if not file_content:
                continue
            task_id += 1
            identifier = method['identifier']
            if not os.path.exists(real_path_cargo[file_path]):
                continue
            flag = retrieve_id(identifier, data[real_path_cargo[file_path].replace("repository/", "/root/")])
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
            PASS = False
            COMPILE_PASS = False
            jump_flag = False
            for idx in tqdm(range(num_return_sequences), colour='yellow'):
                mu = None
                tilde = None
                gas = None
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
                match_path = real_path_cargo[file_path].split('/')[-1]
                cwd_key = "/".join(file_path.split('/')[0:3])
                test_process = subprocess.run(['forge', 'test', '--match-path', f'{match_path}'],
                                              capture_output=True, cwd=cwd_dir_cargo[cwd_key], timeout=120)
                captured_stdout = test_process.stdout.decode()
                with open(f"{file_path}", 'w') as f:
                    f.write(source_bk)
                if "Compiler run failed:" in captured_stdout:
                    logger.error("captured_stdout:\n" + captured_stdout)
                    log_dict[
                        f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}"] \
                        = {'file_path': file_path, 'real_file_path': real_path_cargo[file_path],
                           'COMPILE_PASS': False, 'PASS': False,
                           'patch': patch_st, 'human_labeled_comment': comment, 'source_p': source[start:end],
                           'Compile_ERROR_Message': captured_stdout, 'FAIL_Message': None,
                           'patch_length': patch_length, 'GAS': None, 'patch_pretty': patch}
                    task_total[str(task_id)] += 1
                    continue
                elif "No tests match the provided pattern:" in captured_stdout:
                    jump_flag = False
                    logger.error("captured_stdout:\n" + captured_stdout)
                    continue
                logger.info_white("captured_stdout:\n" + captured_stdout)
                COMPILE_PASS = COMPILE_PASS or True
                pattern = re.compile(
                    r'Ran\s+(-?\d+)\s+test\s+suites?\s+in\s+([\d.]+)\s*(ms|s|µs)\s+'
                    r'\(([\d.]+)\s*(ms|s|µs)\s+CPU time\):\s+'
                    r'(\d+)\s+tests passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped\s+\((\d+)\s+total tests\)'
                )
                lines = captured_stdout.strip().split('\n')
                result_dict = {}
                for line in lines:
                    if '[PASS]' in line:
                        parts = line.split(' (runs: ')
                        gas_parts = line.split(' (gas: ')
                        if len(parts) >= 2:
                            test_part = parts[0].split('[PASS] ')[1]
                            index = test_part.find('(')
                            function_name = test_part[:index] if index != -1 else test_part
                            runs_part = parts[1]
                            mu_match = re.search(r'μ:\s*(\d+)', runs_part)
                            tilde_match = re.search(r'~:\s*(\d+)', runs_part)

                            if mu_match and tilde_match:
                                mu = int(mu_match.group(1))
                                tilde = int(tilde_match.group(1))
                        elif len(gas_parts) >= 2:
                            test_part = parts[0].split('[PASS] ')[1]
                            index = test_part.find('(')
                            function_name = test_part[:index] if index != -1 else test_part
                            gas = int(gas_parts[1][:-2])
                        else:
                            raise NotImplementedError("Unknown test output format!!! Please check the gas matching part of the code!!!")
                        result_dict[function_name] = {'μ': mu, '~': tilde, 'gas': gas}
                matches = pattern.findall(captured_stdout)
                try:
                    passes = int(matches[-1][5])
                    fails = int(matches[-1][6])
                    skips = int(matches[-1][7])
                    total = int(matches[-1][8])
                    PASS = PASS or True if fails == 0 else PASS or False
                    logger.info_white("============================")
                    logger.info_white("PASS: " + str(PASS))
                    logger.info_white("passes: " + str(passes))
                    logger.info_white("failures: " + str(fails))
                    logger.info_white("skips: " + str(skips))
                    logger.info_white("total: " + str(total))
                    task_total[str(task_id)] += 1
                    task_compiled_correct[str(task_id)] += 1
                    if fails == 0:
                        task_correct[str(task_id)] += 1
                except Exception as e:
                    logger.error("Error: " + str(e))
                    logger.error("captured_stdout:\n" + captured_stdout)
                    continue
                log_dict[
                    f"patch/{rag_path}/{args.model}_shot_{args.shot}_{context}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}"] \
                    = {'file_path': file_path, 'real_file_path': real_path_cargo[file_path],
                       'COMPILE_PASS': COMPILE_PASS, 'PASS': PASS,
                       'patch': patch_st, 'human_labeled_comment': comment, 'source_p': source[start:end],
                       'Compile_ERROR_Message': None, 'FAIL_Message': None if PASS else captured_stdout,
                       'patch_length': patch_length, 'identifier': identifier, 'GAS': result_dict, 'patch_pretty': patch}
            if not jump_flag:
                logger.error("jumping due to jump_flag is False!!!")
                continue
            with open(filename, 'w') as f:
                json.dump(serialize(log_dict), f, indent=4)
            num_samples_list = list(task_total.values())
            num_correct_list = [task_correct.get(file_path, 0) for file_path in task_total.keys()]
            num_compile_correct_list = [task_compiled_correct.get(file_path, 0) for file_path in task_total.keys()]
            k = [1, 5, 10, ]
            # filter k values that are larger than num_return_sequences
            k = [current_k for current_k in k if current_k <= num_return_sequences]
            pass_at_k_values = {}
            compile_at_k_values = {}
            for current_k in k:
                if all(n >= current_k for n in num_samples_list):
                    pass_at_k = estimate_pass_at_k(num_samples_list, num_correct_list, current_k).mean()
                    compile_at_k = estimate_pass_at_k(num_samples_list, num_compile_correct_list, current_k).mean()
                else:
                    eligible = [(c, n) for c, n in zip(num_correct_list, num_samples_list) if n >= current_k]
                    compile_eligible = [(c, n) for c, n in zip(num_compile_correct_list, num_samples_list)
                                        if n >= current_k]
                    if eligible:
                        pass_at_k = sum(1.0 if c >= 1 else 0.0 for c, n in eligible) \
                                    / len(eligible)
                    else:
                        pass_at_k = 0.0
                    if compile_eligible:
                        compile_at_k = sum(1.0 if c >= 1 else 0.0 for c, n in compile_eligible) \
                                       / len(compile_eligible)
                    else:
                        compile_at_k = 0.0
                pass_at_k_values[f"pass@{current_k}"] = pass_at_k
                compile_at_k_values[f"compile@{current_k}"] = compile_at_k

            number_pass = sum(num_correct_list)
            number_total = sum(num_samples_list)
            number_fail = number_total - number_pass
            number_compiled_total += 1
            number_compiled_fail += 1 if not COMPILE_PASS else 0
            logger.info("-----------------------------")
            for key, value in pass_at_k_values.items():
                logger.info(f"{key}: {value}")
            logger.info("-*-**-*-**-*-**-*-**-*-**-*-*")
            for key, value in compile_at_k_values.items():
                logger.info(f"{key}: {value}")
            compile_success_rate = (number_compiled_total - number_compiled_fail) \
                                   / number_compiled_total if number_compiled_total > 0 else 0.0
            logger.info(f"COMPILE successful rate: {compile_success_rate}")
