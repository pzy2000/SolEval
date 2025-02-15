import os
from datetime import datetime
from colorama import init, Fore, Style
init(autoreset=True)


class MyLogger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        log_dir = os.path.dirname(os.path.abspath(self.logfile_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _write_to_file(self, level, message):
        with open(self.logfile_path, 'a', encoding='utf-8') as f:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{time_str}] [{level}] {message}\n")

    def _print_to_console(self, color, level, message):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{color}[{time_str}] [{level}] {message}{Style.RESET_ALL}")

    def info(self, message):
        self._write_to_file("INFO", message)
        self._print_to_console(Fore.GREEN, "INFO", message)

    def info_white(self, message):
        self._write_to_file("INFO", message)
        self._print_to_console(Fore.WHITE, "INFO", message)

    def info_blue(self, message):
        self._write_to_file("INFO", message)
        self._print_to_console(Fore.BLUE, "INFO", message)

    def info_green(self, message):
        self._write_to_file("INFO", message)
        self._print_to_console(Fore.GREEN, "INFO", message)

    def warn(self, message):
        self._write_to_file("WARN", message)
        self._print_to_console(Fore.YELLOW, "WARN", message)

    def error(self, message):
        self._write_to_file("ERROR", message)
        self._print_to_console(Fore.RED, "ERROR", message)


class TmpLogger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        log_dir = os.path.dirname(os.path.abspath(self.logfile_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _write_to_file(self, level, message):
        with open(self.logfile_path, 'a', encoding='utf-8') as f:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{time_str}] [{level}] {message}\n")

    def _print_to_console(self, color, level, message):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{color}[{time_str}] [{level}] {message}{Style.RESET_ALL}")

    def log(self, message):
        # self._write_to_file("INFO", message)
        self._print_to_console(Fore.GREEN, "INFO", message)

    def warn(self, message):
        # self._write_to_file("WARN", message)
        self._print_to_console(Fore.YELLOW, "WARN", message)

    def error(self, message):
        # self._write_to_file("ERROR", message)
        self._print_to_console(Fore.RED, "ERROR", message)


if __name__ == "__main__":
    logger = MyLogger("logs/app.log")
    logger.info("This is an info log message.")
    logger.warn("This is a warning log message.")
    logger.error("This is an error log message.")