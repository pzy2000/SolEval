from __future__ import absolute_import, division, print_function
import argparse
import json
import os
import random
import re
import warnings
from datetime import datetime
import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.utils.logger import MyLogger
from tools.utils.replacements import update_id
from tools.utils.retrieve import init_bert_model, query


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_file_with_indentation(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except IOError:
        print(f"Error: An IO error occurred while reading the file {filename}.")
        return None


def few_shot_generation(args, prompt, tokenizer, model, fullid_comment=None):
    output_list = []
    
    if prompt is not None:
        if args.model == "DeepSeek-Coder-V3":
            client = OpenAI(api_key="sk-03f8ceb10b22426bb235639e45aa1c91", base_url="https://api.deepseek.com")
            for i in range(args.sample):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system",
                         "content": prompt},
                        {"role": "user", "content": fullid_comment},
                    ],
                    max_tokens=512,
                    temperature=0.7,
                    stream=False,
                    n=1,
                )
                logger._write_to_file("INFO", "response.usage: " + str(response.usage))
                output = str(response.choices[0].message.content)
                match = re.search(r'\b(function|constructor)\b', output)

                if match:
                    output = output[match.start():]
                else:
                    output = ""
                output = output[:output.rfind("```")]
                output_list.append(output)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            raw_outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=args.p,
                top_k=args.k,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.sample
            )
            for raw_output in raw_outputs:
                logger.info_blue(tokenizer.decode(raw_output))
                output = tokenizer.decode(raw_output[len(inputs[0]):])
                output = output[:output.find("// End")].rstrip()
                output = output[:output.rfind("}") + 1]
                output_list.append(output)
        return output_list


if __name__ == '__main__':
    with open('dataset.json', 'r') as file:
        data = json.load(file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="CodeLlama_7b")
    parser.add_argument('--k',
                        help='The number of highest probability vocabulary tokens to keep '
                             'for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.',
                        type=int, default=50)
    parser.add_argument('--p',
                        help='Only the most probable tokens with probabilities that add up to top_p '
                             'or higher are considered during decoding. The valid range is 0.0 to 1.0. '
                             '1.0 is equivalent to disabled and is the default. Only applies to sampling '
                             'mode. Also known as nucleus sampling.',
                        type=float, default=0.95)
    parser.add_argument('--temperature',
                        help='A value used to warp next-token probabilities in sampling mode. Values less '
                             'than 1.0 sharpen the probability distribution, resulting in "less random" output.'
                             ' Values greater than 1.0 flatten the probability distribution, resulting in "more '
                             'random" output. A value of 1.0 has no effect and is the default. '
                             'The allowed range is 0.0 to 2.0.',
                        type=float, default=1)
    parser.add_argument('--context', action='store_true', default=True, help='Enable context for generation')
    parser.add_argument('--testcase', action='store_true', default=False, help='Enable testcase for generation')
    parser.add_argument('--shot', help='', type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    embedding_list, original_document_list, func_list = init_bert_model()
    # torch.cuda.empty_cache()

    # with open('prompt_template.txt', 'r', encoding='utf-8') as file:
        # prompt_template = file.read()
    log_file = f"log_{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}_{current_time}.txt"
    logger = MyLogger(f"logs_patch/{log_file}")
    if args.model == "DeepSeek-Coder-34B":
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')        
    elif args.model == "DeepSeek-Coder":
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "CodeLlama":
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-34b-Instruct-hf', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-34b-Instruct-hf', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "CodeLlama_7b":
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-7b-Instruct-hf',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    elif args.model == "DeepSeek-Coder-V2":
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "Magicoder":
        tokenizer = AutoTokenizer.from_pretrained('ise-uiuc/Magicoder-S-DS-6.7B', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('ise-uiuc/Magicoder-S-DS-6.7B', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "OpenCode":
        tokenizer = AutoTokenizer.from_pretrained('m-a-p/OpenCodeInterpreter-DS-6.7B', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('m-a-p/OpenCodeInterpreter-DS-6.7B', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    else:
        raise ValueError("Invalid model")

    if not os.path.exists(f"patch/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}"):
        os.makedirs(f"patch/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}")

    import pickle
    real_path_cargo = pickle.load(open("real_path_cargo.pkl", "rb"))
    logger.info("filter Over!")
    for file_path, file_content in tqdm(data.items(), colour='green'):
        if file_path not in real_path_cargo.keys():
            logger.warn("jumping file_path:\n" + file_path)
            continue
        if not file_content:
            continue
        if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
             or "forge" in file_path:
            continue
        logger.info_white("file_path:\n" + file_path)
        for i in range(len(file_content)):
            for method in tqdm(file_content[i]['methods'], colour='yellow'):
                if not file_content[i]['methods']:
                    continue
                identifier = method['identifier']
                if "test/Upgrades.t.sol" in real_path_cargo[file_path]:
                    continue
                flag = update_id(identifier, data[real_path_cargo[file_path]][0])
                if "human_labeled_comment" not in method.keys():
                    continue
                comment = method['human_labeled_comment'].strip() if method['human_labeled_comment'].strip().endswith("\n */") else method['human_labeled_comment'].strip() + "\n */"
                if not comment or not flag:
                    logger.warn("jumping file_path:\n" + file_path + f" for {bool(comment)} comment and {bool(flag)} flag")
                    continue
                if 'context' in method.keys():
                    context = method['context']
                else:
                    raise NotImplementedError("No context")
                context = eval(context)
                if context == set():
                    context = None
                function_full_sig = method['full_signature'].strip() + ' {' + '\n'
                
                examples = query(comment, embedding_list, original_document_list, func_list, args.shot)
                
                for example in examples:
                    prompt = "// Implement the functionality based on the provided requirement\n\n// Requirement\n" + example[0] + "\n\n// Function\n" + example[1] + "\n// End\n\n"
                prompt = prompt + "// Implement the functionality based on the provided requirement\n\n// Requirement\n" + comment + '\n'
                if args.context and context:
                    prompt = prompt + '\n' + "// Context" + '\n'
                    for c in context:
                        prompt = prompt + c + '\n'
                prompt = prompt + '\n' + "// Function" + '\n' + function_full_sig
                if os.path.exists(
                        f"patch/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_4.txt"):
                    logger.info_blue(f"exist patch file, skipping {real_path_cargo[file_path].split('/')[-1]}_function_{identifier}")
                    continue
                logger.warn("prompt: \n")
                logger.warn(prompt)
                logger.warn("\n====================\n")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output_list = few_shot_generation(args, prompt, tokenizer, model, function_full_sig + comment)
                if args.model != "DeepSeek-Coder-V3":
                    output_list = [function_full_sig.strip('\n') + '\n' + output.strip('\n') for output in output_list]
                for idx, out in enumerate(output_list):
                    with open(f"patch/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt", 'w') as f:
                        f.write(out)
