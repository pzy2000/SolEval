import json
from pprint import pprint


# Calculate the percentage of maximum gas gap between the two
def calculate_gas_difference(gas_1, gas_2):
    if gas_1 is None or gas_2 is None:
        return 0
    gas_diff = abs(gas_1 - gas_2)
    max_gas = max(gas_1, gas_2)
    if max_gas == 0:
        return 0
    return (gas_diff / max_gas) * 100


# Extracting gas values from the dictionary (gas, µ, ~)
def extract_gas_values(gas_dict):
    # Extract gas, µ, ~ values, convert to float, return None if value is 'None'
    def safe_convert(value):
        try:
            return float(value) if value != 'None' else None
        except ValueError:
            return None

    gas_value = safe_convert(gas_dict.get('gas', None))
    mu_value = safe_convert(gas_dict.get('μ', None))
    tilde_value = safe_convert(gas_dict.get('~', None))

    return gas_value, mu_value, tilde_value


def extract_func_name(gas_1):
    func_name = []
    for key in gas_1.keys():
        func_name.append(key)
    return func_name


def analyze_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if not isinstance(data, dict):
        print(f"The data was formatted incorrectly, expected to be a dictionary but actually was:{type(data)}")
        return

    data_list = list(data.values())

    if len(data_list) < 5:
        print("Fewer than 5 data entries in the file")
        return
    for i in range(len(data_list) - 4):
        subset = data_list[i:i + 5]

        if all(subset[0]["file_path"] == entry["file_path"] and
               subset[0]["real_file_path"] == entry["real_file_path"] and
               "identifier" in entry.keys() and
               subset[0]["identifier"] == entry["identifier"] and
               entry["COMPILE_PASS"] == "True" and int(entry["patch_length"]) > 5 for entry in subset):

            gas_differences = []
            for j in range(4):
                gas_1 = subset[j]["GAS"]
                gas_2 = subset[j + 1]["GAS"]
                if gas_1 is None or gas_2 is None:
                    continue
                func_name_list_1 = extract_func_name(gas_1)
                func_name_list_2 = extract_func_name(gas_2)
                func_name_list = list(set(func_name_list_1).intersection(set(func_name_list_2)))
                # print("func_name_list:", func_name_list)
                for func_name in func_name_list:
                    # print("func_name:", func_name)
                    gas_1_values = extract_gas_values(gas_1[func_name])
                    gas_2_values = extract_gas_values(gas_2[func_name])
                    # Extract the percentage of disparity per pair of gas (for gas, µ, ~)
                    gas_1_gas, gas_1_mu, gas_1_tilde = gas_1_values
                    gas_2_gas, gas_2_mu, gas_2_tilde = gas_2_values
                    # Calculate and print the gap in μ
                    if gas_1_mu is not None and gas_2_mu is not None:
                        mu_diff = calculate_gas_difference(gas_1_mu, gas_2_mu)
                        if func_name not in gas_incre_rank:
                            gas_incre_rank[func_name] = []
                        gas_incre_rank[func_name].append(mu_diff)
                    # Calculate and print the ~ gap
                    if gas_1_tilde is not None and gas_2_tilde is not None:
                        tilde_diff = calculate_gas_difference(gas_1_tilde, gas_2_tilde)
                        # print(f"~gap percentage: {tilde_diff:.2f}%:")
                        if func_name not in gas_incre_rank:
                            gas_incre_rank[func_name] = []
                        gas_incre_rank[func_name].append(tilde_diff)
                    # print("---")
gas_incre_rank = {}
analyze_jsonl("results_CodeLlama_shot_2_context_True_testcase_False_20250131_003849.jsonl")
# cal average gas_incre_rank by key
print(gas_incre_rank)
for key in gas_incre_rank.keys():
    gas_incre_rank[key] = sum(gas_incre_rank[key]) / len(gas_incre_rank[key])
pprint(gas_incre_rank)
# cal average gas_incre
average_gas_incre = sum(gas_incre_rank.values()) / len(gas_incre_rank)
print("average_gas_incre:", average_gas_incre)
# use Student's t-test to test the significance of the difference
from scipy import stats
import numpy as np
# print("gas_incre_rank:", gas_incre_rank)
gas_incre_rank_values = np.array(list(gas_incre_rank.values()))
print("gas_incre_rank_values:", gas_incre_rank_values)
t, p = stats.ttest_1samp(gas_incre_rank_values, 0)
print("t:", t)
print("p:", p)
