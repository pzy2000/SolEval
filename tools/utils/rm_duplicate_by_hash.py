import hashlib


def calculate_hash(file_path):
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def find_duplicate_solidity_files(roots):
    hash_dict = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.sol'):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        file_hash = calculate_hash(file_path)
                        if file_hash in hash_dict:
                            hash_dict[file_hash].append(file_path)
                        else:
                            hash_dict[file_hash] = [file_path]
                    except Exception as e:
                        print(f"Error extracting {file_path}: {e}")
    duplicates = {hash_val: paths for hash_val, paths in hash_dict.items() if len(paths) > 1}
    return duplicates


import os
from tools.utils.TestParser import TestParser

# change root_path_list to include your dataset
root_path_list = [
    "repository/openzeppelin-contracts",
    "repository/ethernaut/lib/ethernaut.git/contracts",
    "repository/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts-upgradeable",
    "repository/openzeppelin-community-contracts/lib/openzeppelin-community-contracts",
    "repository/uniswap-solidity-hooks-template/lib/uniswap-solidity-hooks-template",
    "repository/openzeppelin-foundry-upgrades/lib/openzeppelin-foundry-upgrades",
    "repository/Account2",
    "repository/solady",
    "repository/forge-std"
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
    for root_path in root_path_list:
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".sol"):
                    full_path = os.path.join(dirpath, filename)
                    sol_files.append(full_path)
    # pprint(sol_files)
    # exit()
    duplicates = find_duplicate_solidity_files(sol_files)
    if duplicates:
        print("Duplicate Solidity file found:")
        for hash_val, paths in duplicates.items():
            print(f"Hash Value: {hash_val}")
            for path in paths:
                print(f"  {path}")
    else:
        print("No duplicate Solidity files were found.")
