import os
import itertools


cwd_dir_cargo = {
    "repository/openzeppelin-contracts": "repository/openzeppelin-contracts",
    "repository/ethernaut": "repository/ethernaut/lib/ethernaut.git/contracts",
    "repository/openzeppelin-contracts-upgradeable": "repository/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts-upgradeable/contracts",
    "repository/openzeppelin-foundry-upgrades": "repository/openzeppelin-foundry-upgrades/lib/openzeppelin-foundry-upgrades",
    "repository/openzeppelin-community-contracts": "repository/openzeppelin-community-contracts/lib/openzeppelin-community-contracts",
    "repository/uniswap-solidity-hooks-template": "repository/uniswap-solidity-hooks-template/lib/uniswap-solidity-hooks-template",
    "repository/Account2": "repository/Account2/lib/Account2",
    "repository/solady": "repository/solady/lib/solady",
    "repository/forge-std": "repository/forge-std/lib/forge-std",
    "repository/openzeppelin-contracts/contracts": "repository/openzeppelin-contracts/contracts",
}

replacements = [
    ("/openzeppelin-contracts/test/", "/openzeppelin-contracts/contracts/"),
    ("/openzeppelin-foundry-upgrades/test/", "/openzeppelin-foundry-upgrades/contracts/"),
    ("/ethernaut.git/contracts/test/", "/ethernaut.git/contracts/src/"),
    (".t.sol", ".sol"),
    ("/test/", "/src/"),
    ("/test/", "/src/utils/"),
    ("/test/utils/", "/contracts/utils/"),
    ("/openzeppelin-community-contracts/test/", "/openzeppelin-community-contracts/contracts/"),
    ("/lib/openzeppelin-contracts-upgradeable/test/utils/",
     "/lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/utils/"),
    ("lib/openzeppelin-contracts-upgradeable/test/",
     "lib/openzeppelin-contracts-upgradeable/lib/openzeppelin-contracts/contracts/"),

]

single_replacements = [
    ("lib/ethernaut.git/contracts/test/metrics/Player.t.sol", "lib/ethernaut.git/contracts/test/utils/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/metrics/Player.t.sol", "lib/ethernaut.git/contracts/src/metrics/Statistics.sol"),
    ("lib/ethernaut.git/contracts/test/metrics/Level.t.sol", "lib/ethernaut.git/contracts/test/utils/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/metrics/Level.t.sol", "lib/ethernaut.git/contracts/src/metrics/Statistics.sol"),
    ("lib/ethernaut.git/contracts/test/metrics/Leaderboard.t.sol", "lib/ethernaut.git/contracts/test/utils/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/metrics/Leaderboard.t.sol",
     "lib/ethernaut.git/contracts/src/metrics/Statistics.sol"),
    ("lib/ethernaut.git/contracts/test/levels/Coinflip.t.sol", "lib/ethernaut.git/contracts/test/utils/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/levels/Coinflip.t.sol",
     "lib/ethernaut.git/contracts/src/attacks/CoinFlipAttack.sol"),
    ("lib/ethernaut.git/contracts/test/levels/Coinflip.t.sol", "lib/ethernaut.git/contracts/src/levels/CoinFlip.sol"),
    ("lib/ethernaut.git/contracts/test/levels/Coinflip.t.sol",
     "lib/ethernaut.git/contracts/src/levels/CoinFlipFactory.sol"),
    ("lib/uniswap-solidity-hooks-template/test/Counter.t.sol", "lib/uniswap-solidity-hooks-template/src/Counter.sol"),
    ("lib/uniswap-solidity-hooks-template/test/Counter.t.sol",
     "lib/uniswap-solidity-hooks-template/lib/v4-core/test/utils/Deployers.sol"),
    ("lib/uniswap-solidity-hooks-template/test/Counter.t.sol",
     "lib/uniswap-solidity-hooks-template/test/utils/EasyPosm.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/openzeppelin-contracts-v4-with-v5-proxies/test/Upgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/openzeppelin-contracts-v4/test/LegacyUpgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/LegacyUpgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/openzeppelin-contracts-v4/test/UnsafeLegacyUpgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/LegacyUpgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/build-info-v2-reference-contract/test/Upgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/build-info-v2-bad/test/Upgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test-profiles/build-info-v2/test/Upgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/UnsafeUpgrades.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/Upgrades.t.sol", "lib/openzeppelin-foundry-upgrades/src/Upgrades.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/internal/Core.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/internal/Core.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/internal/Core.t.sol",
     "lib/openzeppelin-foundry-upgrades/test/internal/StringHelper.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/internal/DefenderDeploy.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/internal/DefenderDeploy.sol"),
    ("lib/openzeppelin-foundry-upgrades/test/internal/Utils.t.sol",
     "lib/openzeppelin-foundry-upgrades/src/internal/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/levels/GatekeeperTwo.t.sol", "lib/ethernaut.git/contracts/test/utils/Utils.sol"),
    ("lib/ethernaut.git/contracts/test/levels/GatekeeperTwo.t.sol",
     "lib/ethernaut.git/contracts/src/attacks/GatekeeperTwoAttack.sol"),
    ("test/token/ERC721/extensions/ERC721Consecutive.t.sol", "contracts/token/ERC721/ERC721.sol"),

]


def update_id(identifier, file_cont):
    flag = False
    for method in file_cont['methods']:
        if identifier in method['body']:
            method['id'].append(identifier)
            flag = True
    return flag


def retrieve_id(identifier, file_cont):
    flag = False
    for method in file_cont:
        for _method in method['id']:
            if identifier in method['id'] and identifier == _method:
                # method['id'].append(identifier)
                # if "pack_1_1" in identifier:
                #     print("=*=" * 10)
                #     print("flag: ", flag)
                #     print("identifier: ", identifier)
                #     print("method['id']: ", method['id'])
                #     print("_method: ", _method)
                #     print("identifier in method['id']: ", identifier in method['id'])
                #     print("identifier == _method: ", identifier == _method)
                #     print("=*=" * 10)
                flag = True
            # elif "pack_1_1" in identifier:
            #     pass
                # print("============================")
                # print("identifier: ", identifier)
                # print("method['id']: ", method['id'])
                # print("_method: ", _method)
                # print("identifier in method['id']: ", identifier in method['id'])
                # print("identifier == _method: ", identifier == _method)
                # print("============================")
    return flag


def update_tdd(test_cont, sol_cont):
    flag = False
    for method in test_cont['methods']:
        for method_sol in sol_cont['methods']:
            if method_sol['identifier'] in method['body']:
                if 'tdd' not in method_sol:
                    method_sol['tdd'] = set()
                method_sol['tdd'].add(method_sol['body'])
                flag = True
    return flag


def generate_replaced_paths(original, replacements, single=False):
    real_file_path_list = []
    num_replacements = len(replacements)

    for r in range(1, num_replacements + 1 if not single else 2):
        for subset in itertools.combinations(replacements, r if not single else 1):
            replaced = original
            for old, new in subset:
                replaced = replaced.replace(old, new)
            if replaced not in real_file_path_list and replaced != original and os.path.exists(replaced):
                real_file_path_list.append(replaced)

    return real_file_path_list


if __name__ == '__main__':
    print("init testing......")
    file_path = "repository/ethernaut/lib/ethernaut.git/contracts/test/metrics/Player.t.sol"
    real_file_path_list = generate_replaced_paths(file_path, replacements)
    print("generated replaced_paths......")
    real_path_cargo = {}
    print("real_file_path_list: ", real_file_path_list)
    if not real_file_path_list:
        print("not found any replaced_paths......")
        print(f"No replacement paths were found: {file_path}")
        a = input("Continue? >>>")
    else:
        for real_file_path in real_file_path_list:
            real_path_cargo[real_file_path] = file_path
            print(f"Found valid path: {real_file_path} Corresponds to original path: {file_path}")
    # print(real_path_cargo)
