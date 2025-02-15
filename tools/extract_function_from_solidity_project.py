import json
import sys
sys.path.append('tools')
sys.path.append('..')
import os
from tqdm import tqdm
from utils.TestParser import TestParser


root_path_list = [
    "repository/openzeppelin-contracts",
    "repository/ethernaut/lib/ethernaut.git/contracts",
    "repository/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts-upgradeable",
    "repository/openzeppelin-community-contracts/lib/openzeppelin-community-contracts",
    "repository/uniswap-solidity-hooks-template/lib/uniswap-solidity-hooks-template",
    "repository/openzeppelin-foundry-upgrades/lib/openzeppelin-foundry-upgrades",
    "repository/Account2",
    "repository/solady",
    "repository/forge-std",
    "repository/murky",
]


def serialize(obj):
    if isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    else:
        return str(obj)


if __name__ == '__main__':
    sol_files = []
    current_dir = os.path.abspath(os.path.dirname(__file__))
    libtree_so_path = os.path.join(current_dir, "../libtree-sitter-solidity.so")

    parser = TestParser(libtree_so_path, "solidity")
    choice = input("Enter 1 to gen from scratch, 2 to continue to merge into parsed_results_with_comment.json")
    if choice == "1":
        pass
    elif choice == "2":
        with open('../rubbish_bin/results_with_context_tdd_repo.json', 'r') as file:
            data = json.load(file)
    # elif choice == "3":
    #     with open('parsed_results.json', 'r') as file:
    #         data = json.load(file)
    #     with open('parsed_results_with_comment.json', 'r') as file:
    #         data.update(json.load(file))
    else:
        raise NotImplementedError("Invalid choice!")
    for root_path in root_path_list:
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".sol"):
                    full_path = os.path.join(dirpath, filename)
                    sol_files.append(full_path)

    parsed_results = {}
    for file_path in tqdm(sol_files, desc="Parsing .sol files"):
        if file_path.startswith("repository/openzeppelin-contracts/lib"):
            continue
        if file_path.startswith("repository/ethernaut/lib/ethernaut.git/contracts/lib"):
            continue
        if file_path.startswith("repository/openzeppelin-community-contracts/lib/openzeppelin-community-contracts/lib"):
            continue
        if file_path.startswith("repository/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts-upgradeable/lib"):
            continue
        if file_path.startswith("repository/uniswap-solidity-hooks-template/lib/uniswap-solidity-hooks-template/lib"):
            continue
        if file_path.startswith("repository/openzeppelin-foundry-upgrades/lib/openzeppelin-foundry-upgrades/lib"):
            continue
        if file_path.startswith("repository/Account2/lib/Account2/lib"):
            continue
        parsed_classes = parser.parse_file(file_path)
        parsed_results[file_path] = parsed_classes
    if choice == "1":
        output_json_file = "repository/SolParser/parsed_results.json"
        with open(output_json_file, "w") as json_file:
            json.dump(serialize(parsed_results), json_file, indent=4)
    elif choice == "2":
        data.update(parsed_results)
        output_json_file = "repository/SolParser/parsed_results_with_context_new.json"
        with open(output_json_file, "w") as json_file:
            json.dump(serialize(data), json_file, indent=4)
    else:
        raise NotImplementedError("Invalid choice!")
    print(f"Parsed results have been exported to {output_json_file} with mode {choice}")
