from __future__ import absolute_import, division, print_function
import re
import subprocess
import warnings
from datetime import datetime
from utils.logger import MyLogger
from utils.replacements import cwd_dir_cargo

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    log_dict = {}
    log_file = f"log_TEST_{current_time}.txt"
    logger = MyLogger(f"logs/{log_file}")
    for cwd_key in cwd_dir_cargo.keys():
        logger.info_white(f"cwd: {cwd_dir_cargo[cwd_key]}")
        logger.info_white(f"start to test {cwd_key}")
        logger.info_white("========================================")
        test_process = subprocess.run(['forge', 'test'],
                                      capture_output=True, cwd=cwd_dir_cargo[cwd_key], timeout=120)
        captured_stdout = test_process.stdout.decode()
        if "Compiler run failed:" in captured_stdout:
            logger.error("captured_stdout:\n" + captured_stdout)
            exit(-1)
        logger.info_white("captured_stdout:\n" + captured_stdout)
        pattern = re.compile(
            r'Ran\s+(-?\d+)\s+test\s+suites?\s+in\s+([\d.]+)\s*(ms|s|µs)\s+'
            r'\(([\d.]+)\s*(ms|s|µs)\s+CPU time\):\s+'
            r'(\d+)\s+tests passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped\s+\((\d+)\s+total tests\)'
        )
        lines = captured_stdout.strip().split('\n')
        result_dict = {}
        matches = pattern.findall(captured_stdout)
        # print("matches:", matches)
        # print("length:", len(matches))
        try:
            passes = int(matches[-1][5])
            fails = int(matches[-1][6])
            skips = int(matches[-1][7])
            total = int(matches[-1][8])
            if fails > 0:
                logger.error(f"fails: {fails}")
                logger.error("Error project: " + cwd_key)
                exit(-1)
        except Exception as e:
            logger.error("captured_stdout:\n" + captured_stdout)
            logger.error("Error: " + str(e))
            logger.error("Error project: " + cwd_key)
            exit(-1)
