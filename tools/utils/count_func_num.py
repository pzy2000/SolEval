import json


if __name__ == '__main__':
    with open('raw_data.json', 'r') as file:
        data = json.load(file)
    func_num = 0
    jump_num = 0
    func_cargo = {}
    avg_lines_of_func = 0
    lines_of_func_cargo = {}
    for file_path, file_content in data.items():
        cwd_key = "/".join(file_path.split('/')[2:3])
        for i in range(len(file_content)):
            if not file_content[i]['methods']:
                continue
            for method in file_content[i]['methods']:
                # if "human_labeled_comment" in method.keys():
                #     continue
                if not method["full_signature"].startswith("function"):
                    continue
                if method["testcase"] == "True":
                    continue
                func_num += 1
                if cwd_key not in func_cargo:
                    func_cargo[cwd_key] = []
                func_cargo[cwd_key].append(method["full_signature"])
                lines_of_func = int(method["end"]) - int(method["start"]) - 2
                if cwd_key not in lines_of_func_cargo:
                    lines_of_func_cargo[cwd_key] = []
                lines_of_func_cargo[cwd_key].append(lines_of_func)
    # count the number of functions by project
    for key, value in func_cargo.items():
        print(key, len(value))
    print("============================================")
    # count the average number of lines of functions by project
    for key, value in lines_of_func_cargo.items():
        # keep 2 decimal places
        avg_lines_of_func += sum(value) / len(value)
        print(key, round(sum(value) / len(value), 2))
    print(func_num)
