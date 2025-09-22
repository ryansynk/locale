import os
import subprocess
from tqdm import tqdm

def run_art_on_all_files():
    """
    Traverses train, val, and test directories to run the art_exe command
    on every 'query.fa' file found.
    """
    # List of the base directories to search through
    base_directories = ['../data/dataset/train', '../data/dataset/val', '../data/dataset/test']

    # Get the absolute path to the executable. This script assumes it is
    # run from the same directory as 'art_exe' and the train/val/test folders.
    executable_path = os.path.abspath("../data/tools/art_bin_MountRainier/art_illumina")

    if not os.path.isfile(executable_path):
        print(f"Error: Executable not found at '{executable_path}'")
        print("Please ensure you run this script from the same directory as 'art_exe'.")
        return

    # Loop through each base directory (train, val, test)
    for base_dir in base_directories:
        # Check if the base directory exists
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found. Skipping.")
            continue

        print(f"\n--- Processing directory: {base_dir} ---")

        # os.walk recursively visits every directory and subdirectory
        for dirpath, _, filenames in tqdm(os.walk(base_dir)):
            # Check if 'query.fa' exists in the current directory
            if 'query.fa' in filenames:
                input_file_path = os.path.join(dirpath, 'query.fa')

                # The command needs to be run with the subdirectory as the current
                # working directory so that './query.fa' and './paired_end_com' resolve correctly.
                # We calculate the relative path from the subdirectory back to the executable.
                rel_executable_path = os.path.relpath(executable_path, start=dirpath)

                # Define the command and its arguments as a list
                command = [
                    rel_executable_path,
                    "-ss", "HS25",
                    "-i", "./query.fa",
                    "-o", "./paired_end_com",
                    "-l", "100",
                    "-f", "30",
                    "-p",
                    "-m", "400",
                    "-s", "40",
                    "-na"
                ]

                try:
                    # Execute the command from within the subdirectory
                    subprocess.run(
                        command,
                        cwd=dirpath,  # Set the current working directory
                        check=True,   # Raise an exception if the command fails
                        capture_output=True, # Capture stdout and stderr
                        text=True     # Decode stdout/stderr as text
                    )

                except FileNotFoundError:
                    print(f"Error: Command not found. Is '{rel_executable_path}' correct?")
                    # This might happen if the executable path logic is flawed.
                    break
                except subprocess.CalledProcessError as e:
                    # This error is raised if the command returns a non-zero exit code
                    print(f"Error running command on {input_file_path}")
                    print(f"Return code: {e.returncode}")
                    print(f"Stdout: {e.stdout}")
                    print(f"Stderr: {e.stderr}\n")
                    continue # Continue to the next file even if one fails

    print("--- Script finished ---")

if __name__ == "__main__":
    run_art_on_all_files()

