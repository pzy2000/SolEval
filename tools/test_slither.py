import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tools.utils.logger import MyLogger
from utils.replacements import cwd_dir_cargo
import time


def get_solc_version(contract_path):
    with open(contract_path, "r", encoding="utf-8") as f:
        content = f.read()

    version_match = re.search(r"pragma\s+solidity\s+([^;]+);", content)
    if not version_match:
        return None

    version_spec = version_match.group(1)

    if version_spec.startswith("^"):
        base_version = version_spec[1:].split(".")  # 0.8.0 → 0.8.x
        return f"{base_version[0]}.{base_version[1]}.{base_version[2]}"
    return version_spec


def install_solc_version(version):
    try:
        result = subprocess.run(
            ["solc-select", "install", version],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        if "Installing" in result.stdout:
            return True
        print(f"Install failure: {result.stderr}")
        return False
    except Exception as e:
        print(f"Install Error: {str(e)}")
        return False


def switch_solc_version(target_version):
    installed_versions = subprocess.run(
        ["solc-select", "versions"],
        stdout=subprocess.PIPE,
        text=True
    ).stdout
    # print("installed_versions:", installed_versions)
    # print("target_version:", target_version)
    if target_version not in installed_versions:
        print(f"version {target_version} is not installed, please try to install automatically...")
        if not install_solc_version(target_version):
            return False

    try:
        result = subprocess.run(
            ["solc-select", "use", target_version],
            capture_output=True,
            text=True,
            timeout=15
        )
        if "Switched" in result.stdout:
            return True
        print(f"Switch Fail: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("Switch Timeout")
        return False


# def get_allow_paths(contract_path):
#     path = Path(contract_path)
#     contracts_dir = path.parent.absolute()
#     interfaces_dir = contracts_dir.parent / "interfaces"
#     utils_dir = contracts_dir.parent / "utils"
#
#     return f".,{contracts_dir},{interfaces_dir},{utils_dir}"


def safe_get_subdirs(root_path, max_depth=5):
    allowed_dirs = [str(Path(root_path).absolute())]
    # print("allowed_dirs:", allowed_dirs)
    current_level = [Path(root_path)]
    # print("current_level:", current_level)
    for depth in range(max_depth):
        next_level = []
        for d in current_level:
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        allowed_dirs.append(str(child.absolute()))
                        next_level.append(child)
            except PermissionError:
                continue
        current_level = next_level

    sorted_allowed_dirs = sorted(list(set(allowed_dirs)))
    return "., /root"


# allowed_paths = safe_get_subdirs("/root", max_depth=5)
def analyze_contract(contract_path):
    # print("allow_paths:", allow_paths)
    # exit()
    remappings = "@openzeppelin/=node_modules/@openzeppelin/"

    start_time = time.time()
    required_version = get_solc_version(contract_path)
    if not required_version:
        return {"error": True, "message": "Unable to parse Solidity version"}
    #
    # installed_versions = subprocess.run(
    #     ["solc-select", "versions"],
    #     stdout=subprocess.PIPE,
    #     text=True
    # ).stdout
    # available_versions = subprocess.run(
    #     ["solc-select", "install"],
    #     stdout=subprocess.PIPE,
    #     text=True
    # ).stdout
    # # print("available_versions:", available_versions)
    # # print("available_versions:", installed_versions)
    # if required_version not in available_versions and required_version not in installed_versions:
    if not switch_solc_version(required_version):
        return {"error": True, "message": f"Unable to switch to {required_version}"}
    try:
        version_check = subprocess.run(
            ["solc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        end_time = time.time()
        print("Time cost:", end_time - start_time)
        try:
            report_file = Path("slither_report.json")

            cmd = [
                "slither",
                contract_path,
                "--json",
                str(report_file),
                "--disable-color",
                # "--allow-paths", allow_paths,
                "--solc-remaps", remappings,
                "--exclude-informational"
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60
            )
            # print("result:", result)
            # print("=====================================")
            # print("return code:", result.returncode)
            # print("=====================================")
            if result.returncode == 1:
                return {
                    "error": True,
                    "message": f"Slither execute failure: {result.stderr}"
                }

            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
                report_file.unlink()
                return parse_report(report)
            else:
                return {"error": True, "message": "Report file not found"}

        except Exception as e:
            return {"error": True, "message": str(e)}

    except Exception as e:
        return {"error": True, "message": str(e)}


def parse_report(report):
    findings = {
        "high": [],
        "medium": [],
        "low": [],
        "informational": [],
        "high_num": 0,
    }
    VUL = False
    for detector in report.get("results", {}).get("detectors", []):
        finding = {
            "type": detector["check"],
            "description": detector["description"],
            "impact": detector["impact"].lower(),
            "confidence": detector["confidence"].lower(),
            "elements": detector["elements"]
        }
        severity = detector["impact"].lower()
        if severity not in ["informational", "optimization"]:
            if severity == "high" and detector["confidence"].lower() == "high":
                findings["high_num"] += 1
                VUL = True
            findings[severity].append(finding)
    return {
        "error": False,
        "vul_message": findings,
        "vul": VUL,
        "compiler_version": report.get("compiler", {}).get("version"),
        "contract_name": report.get("contracts", [])[0] if report.get("contracts") else None
    }


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"log_{current_time}.txt"
    logger = MyLogger(f"logs_filter/{log_file}")
    with open('results_with_context_666.json', 'r') as file:
        data = json.load(file)
    import pickle
    real_path_cargo = pickle.load(open("real_path_cargo.pkl", "rb"))
    result = {}
    for cwd_key in tqdm(cwd_dir_cargo.keys(), colour='green'):
        file_path = cwd_dir_cargo[cwd_key]
        logger.info_white("file_path:\n" + file_path)
        contract_file = file_path
        analysis = analyze_contract(contract_file)

        if analysis.get("error"):
            print(f"Analyze fail: {analysis['message']}")
        else:
            print("Analyze success!")
            print(f"Use compiler: {analysis['compiler_version']}")
        result[file_path] = analysis
    # export result to a json file
        with open('rubbish_bin/ALL_slither.json', 'w') as file:
            json.dump(result, file)


