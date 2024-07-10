import sys
import select
from repromodel_core.src.utils import delete_command_outputs, print_to_file

def log_message(message, level="INFO"):
    if "werkzeug" not in message and "HTTP" not in message: #filtering out backend pings
        #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #timestamp already implemented in utils 
        print_to_file(f"[{level}] {message}")
        sys.__stdout__.flush()  # Ensure immediate flush for stdout

def main():
    delete_command_outputs()
    while True:
        inputs = [sys.stdin]
        readable, _, _ = select.select(inputs, [], [])

        for stream in readable:
            line = stream.readline()
            if not line:
                log_message("No more input, exiting", "DEBUG")
                return

            log_message(line.strip(), "INFO")

if __name__ == "__main__":
    main()