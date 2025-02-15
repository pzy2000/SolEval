import json
import pickle
from datetime import datetime
from tqdm import tqdm
from tools.utils.logger import MyLogger


if __name__ == '__main__':
    with open('results_with_context_tdd_repo.json', 'r') as file:
        data = json.load(file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"log_{current_time}.txt"
    logger = MyLogger(f"logs_filter/{log_file}")
    real_path_cargo = pickle.load(open("real_path_cargo.pkl", "rb"))
    logger.info("filter Over!")
    filtered_data = {}
    count = 0
    for file_path, file_content in tqdm(data.items(), colour='green'):
        if file_path not in real_path_cargo.keys():
            logger.warn("jumping file_path:\n" + file_path)
            continue
        if not file_content:
            continue
        # if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") or "forge" in file_path:
        #     continue
        logger.info_white("file_path:\n" + file_path)
        filtered_file_content = []
        for i in range(len(file_content)):
            filtered_methods = []
            for method in tqdm(file_content[i]['methods'], colour='yellow'):
                # if not file_content[i]['methods']:
                #     continue
                identifier = method['identifier']
                # if "test/Upgrades.t.sol" in real_path_cargo[file_path]:
                #     continue
                if "human_labeled_comment" in method.keys():
                    continue
                # print(method["body"])
                # print("=====================================")
                filtered_methods.append(method)
                count += 1
            if filtered_methods:
                filtered_file_content.append({'methods': filtered_methods})
        if filtered_file_content:
            filtered_data[file_path] = filtered_file_content
    output_file = "example.json"
    # print("len(filtered_data): ", len(filtered_data))
    print("count: ", count)
    with open(output_file, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)
    logger.info(f"Filtered data has been saved to {output_file}")
