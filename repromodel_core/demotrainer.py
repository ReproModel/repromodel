import time
import os

def update_progress_bar(file_path, total_time=10, update_interval=2):
    progress_length = 50  # Length of the progress bar
    num_updates = total_time // update_interval
    
    with open(file_path, 'w') as file:
        for i in range(num_updates + 1):
            progress = int((i / num_updates) * progress_length)
            progress_bar = '[' + '#' * progress + ' ' * (progress_length - progress) + ']'
            file.write(f"\r{progress_bar} {int((i / num_updates) * 100)}%\n")
            file.flush()
            time.sleep(update_interval)

def main():
    # Get the path of the script itself
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)

    # Define the relative path
    relative_path = "test_logs/Training_logs/progress.txt"

    # Combine the script's directory with the relative path to get the absolute path
    absolute_path = os.path.join(script_dir, relative_path)


    file_path = absolute_path
    total_time = 30
    update_interval = 1
    update_progress_bar(file_path, total_time, update_interval)

if __name__ == "__main__":
    main()
