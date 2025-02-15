import sys
from pprint import pprint

sys.path.append('..')
import copy
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)


# if torch.cuda.device_count() > 1:
# print(f"Using {torch.cuda.device_count()} GPUs!")
# model = nn.DataParallel(model)

def init_bert_model():
    print("init bert model...")
    document_list = []
    func_list = []
    task_id = 0
    with open('data/example.json', 'r') as file:
        data = json.load(file)
    for file_path, file_content in tqdm(data.items(), desc="Loading document_list..."):
        for method in file_content:
            task_id += 1
            identifier = method['identifier']
            human_labeled_comment = method['human_labeled_comment'].strip() if method[
                'human_labeled_comment'].strip().endswith("\n */") else method[
                                                                            'human_labeled_comment'].strip() + "\n */"
            document_list.append(human_labeled_comment)
            func_list.append(method)
    original_document_list = copy.deepcopy(document_list)
    if os.path.exists('../prebuilt/cls_embeddings.npy'):
        cls_embeddings = np.load('../prebuilt/cls_embeddings.npy')
        cls_embeddings = torch.tensor(cls_embeddings).to(device)
    else:
        input_ids = tokenizer(original_document_list, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**input_ids)
        last_hidden_states = outputs.last_hidden_state
        cls_embeddings = last_hidden_states[:, 0, :]
        np.save('prebuilt/cls_embeddings.npy', cls_embeddings.cpu().numpy())
    torch.cuda.empty_cache()
    # from numba import cuda
    # device = cuda.get_current_device()
    # device.reset()
    return cls_embeddings, original_document_list, func_list


def query(input_requirements: str, embedding_list: list, original_document_list: list, func_list: list, k: int) -> list:
    query = input_requirements
    query_ids = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**query_ids)
    last_hidden_states = outputs.last_hidden_state
    query_embeddings = last_hidden_states[:, 0, :]
    cosine_similarity = torch.nn.functional.cosine_similarity(query_embeddings, embedding_list, dim=1)
    _, topk_indices = torch.topk(cosine_similarity, k)
    result = []
    for index in topk_indices:
        original_doc = original_document_list[index]
        func_dict = func_list[index]
        result.append([original_doc, func_dict["body"],
                       func_dict["context"] if func_dict["context"] != "set()" else "No context for this function"])
    torch.cuda.empty_cache()
    return result


def query_random(original_document_list: list, func_list: list, k: int) -> list:
    # seed = 666
    # random.seed(seed)
    # torch.manual_seed(seed)
    total_items = len(original_document_list)
    random_indices = random.sample(range(total_items), k)
    # print("choose random index:", random_indices)
    result = []
    for index in random_indices:
        original_doc = original_document_list[index]
        func_dict: dict = func_list[index]
        result.append([original_doc, func_dict["body"],
                       func_dict["context"] if func_dict["context"] != "set()" else "No context for this function"])
    return result


if __name__ == '__main__':
    embedding_list, original_document_list, func_list = init_bert_model()
    inpu_test = "/**\n * @notice Packs a `bytes2` and a `bytes10` into a single `bytes12` value.\n *\n * @dev This function uses inline assembly to perform bitwise operations to combine the two input bytes.\n * - The `bytes2` value is shifted left by 240 bits and masked to ensure it occupies the correct position.\n * - The `bytes10` value is shifted left by 176 bits and masked to ensure it occupies the correct position.\n * - The two values are then combined using a bitwise OR operation.\n *\n * @param left The `bytes2` value to be packed into the higher-order bits of the result.\n * @param right The `bytes10` value to be packed into the lower-order bits of the result.\n * @return result The combined `bytes12` value containing both `left` and `right`.\n *\n * Steps:\n * 1. Mask and shift the `left` value to align it with the higher-order bits of the result.\n * 2. Mask and shift the `right` value to align it with the lower-order bits of the result.\n * 3. Combine the two values using a bitwise OR operation to produce the final `bytes12` result.\n"
    k = 2
    pprint(query(inpu_test, embedding_list, original_document_list, func_list, k))
