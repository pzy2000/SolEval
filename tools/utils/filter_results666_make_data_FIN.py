import json
from datetime import datetime
import pickle
from tqdm import tqdm
from tools.utils.logger import MyLogger
from tools.utils.replacements import update_id

if __name__ == '__main__':
    with open('results_with_context_tdd_repo.json', 'r') as file:
        data = json.load(file)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"log_{current_time}.txt"
    logger = MyLogger(f"logs_filter/{log_file}")
    real_path_cargo = pickle.load(open("real_path_cargo.pkl", "rb"))
    logger.info("filter Over!")

    filtered_data = {}

    for file_path, file_content in tqdm(data.items(), colour='green'):
        if file_path not in real_path_cargo.keys():
            logger.warn("jumping file_path:\n" + file_path)
            continue

        if not file_content:
            continue

        if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") or "forge" in file_path:
            continue

        logger.info_white("file_path:\n" + file_path)

        filtered_file_content = []

        for i in range(len(file_content)):
            filtered_methods = []
            for method in tqdm(file_content[i]['methods'], colour='yellow'):
                if int(method["end"]) - int(method["start"]) <= 2:
                    continue
                if not file_content[i]['methods']:
                    continue

                identifier = method['identifier']
                if "test/Upgrades.t.sol" in real_path_cargo[file_path]:
                    continue

                flag = update_id(identifier, data[real_path_cargo[file_path]][0])

                if "human_labeled_comment" not in method.keys():
                    continue

                comment = method['human_labeled_comment'].strip() if method['human_labeled_comment'].strip().endswith("\n */") else method[
                                                                                                                    'human_labeled_comment'].strip() + "\n */"

                if not comment or not flag:
                    logger.warn(
                        "jumping file_path:\n" + file_path + f" for {bool(comment)} comment and {bool(flag)} flag")
                    continue

                print(method["body"])
                print("=====================================")
                filtered_methods.append(method)

            if filtered_methods:
                filtered_file_content.append({'methods': filtered_methods})

        if filtered_file_content:
            filtered_data[file_path] = filtered_file_content

    output_file = "dataset.json"
    with open(output_file, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

    logger.info(f"Filtered data has been saved to {output_file}")
