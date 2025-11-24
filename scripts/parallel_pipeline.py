#!/usr/bin/env python3

import subprocess
import os
import argparse
import sys
import logging
import time
from multiprocessing import Pool
from shlex import quote

# --- Configuration for Logging ---
# This setup will print logs to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def process_accession(task_args):
    """
    Worker function to process a single accession.
    This function runs the full aws -> zstdcat -> minimap2 -> awk pipeline.
    """
    # 1. Unpack all arguments for this task
    # We pass the full 'args' namespace to keep it simple
    accession, args = task_args

    logging.info(f"Starting job for {accession}...")

    # 2. Define file paths
    output_file = os.path.join(args.output_dir, f"{accession}_pairs.txt")
    s3_path = f"s3://logan-pub/c/{accession}/{accession}.contigs.fa.zst"

    # 3. Define the command pipelines as lists
    cmd_aws = [args.aws_bin, "s3", "cp", s3_path, "-", "--no-sign-request"]

    cmd_zstd = [args.zstdcat_bin]

    # --- IMPORTANT FIX ---
    # The original script used -a (SAM output) but the awk script
    # ($10/$11) is for PAF format. We remove -a to get PAF.
    # We use '-' to tell minimap2 to read the TARGET from stdin.
    cmd_map = [
        args.minimap_bin,
        "-x",
        "asm20",
        "-t",
        str(args.threads_per_job),
        args.transcripts_file,  # Read reference (transcripts) from file
        "-",  # Read target (contigs) from stdin
    ]

    # The awk script that expects PAF format
    awk_script = '$10 / $11 > 0.9 && $12 >= 30 {print $1"\t"$6}'
    cmd_awk = [args.awk_bin, awk_script]

    # 4. Execute the pipeline
    try:
        # Open the final output file for writing
        with open(output_file, "w") as f_out:

            # Start the processes and pipe them together
            # p_aws -> p_zstd -> p_map -> p_awk -> f_out

            p_aws = subprocess.Popen(
                cmd_aws, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            p_zstd = subprocess.Popen(
                cmd_zstd,
                stdin=p_aws.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p_aws.stdout.close()  # Allow p_aws to receive SIGPIPE

            p_map = subprocess.Popen(
                cmd_map,
                stdin=p_zstd.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p_zstd.stdout.close()  # Allow p_zstd to receive SIGPIPE

            p_awk = subprocess.Popen(
                cmd_awk, stdin=p_map.stdout, stdout=f_out, stderr=subprocess.PIPE
            )
            p_map.stdout.close()  # Allow p_map to receive SIGPIPE

            # 5. Wait and collect stderr from all components
            # This helps debug which part of the pipe failed
            aws_err = p_aws.stderr.read().decode()
            zstd_err = p_zstd.stderr.read().decode()
            map_err = p_map.stderr.read().decode()
            awk_err = p_awk.stderr.read().decode()

            # Wait for processes to terminate
            p_aws.wait()
            p_zstd.wait()
            p_map.wait()
            p_awk.wait()

            # 6. Check return codes
            if p_aws.returncode != 0:
                logging.error(
                    f"Job {accession} FAILED: aws-cli failed (Code {p_aws.returncode}): {aws_err}"
                )
                return False
            if p_zstd.returncode != 0:
                logging.error(
                    f"Job {accession} FAILED: zstdcat failed (Code {p_zstd.returncode}): {zstd_err}"
                )
                return False
            if p_map.returncode != 0:
                logging.error(
                    f"Job {accession} FAILED: minimap2 failed (Code {p_map.returncode}): {map_err}"
                )
                return False
            if p_awk.returncode != 0:
                logging.error(
                    f"Job {accession} FAILED: awk failed (Code {p_awk.returncode}): {awk_err}"
                )
                return False

        logging.info(f"Successfully finished job for {accession}.")
        return True

    except Exception as e:
        logging.error(f"Job {accession} FAILED with critical error: {e}")
        # Clean up partial file on error
        if os.path.exists(output_file):
            os.remove(output_file)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run minimap2 in parallel on SRA accessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required Arguments ---
    parser.add_argument(
        "--accessions_file",
        required=True,
        help="Path to text file with one accession per line.",
    )
    parser.add_argument("--minimap_bin", required=True, help="Path to minimap2 binary.")
    parser.add_argument(
        "--transcripts_file",
        required=True,
        help="Path to gencode transcripts file (e.g., gencode.v49.transcripts).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store output <acc>_pairs.txt files.",
    )

    # --- Optional Arguments for Parallelism ---
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (accessions) to run.",
    )
    parser.add_argument(
        "-t",
        "--threads_per_job",
        type=int,
        default=3,
        help="Number of threads for *each* minimap2 job (-t flag).",
    )

    # --- Optional Arguments for Binary Paths ---
    parser.add_argument("--aws_bin", default="aws", help="Path to aws CLI binary.")
    parser.add_argument(
        "--zstdcat_bin", default="zstdcat", help="Path to zstdcat binary."
    )
    parser.add_argument("--awk_bin", default="awk", help="Path to awk binary.")

    args = parser.parse_args()

    # --- 1. Setup ---
    logging.info(
        f"Starting run with {args.jobs} parallel jobs ({args.threads_per_job} threads each)."
    )
    logging.info(f"Total max threads: {args.jobs * args.threads_per_job}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Load Accessions ---
    try:
        with open(args.accessions_file, "r") as f:
            # Strip whitespace and remove any empty lines
            accessions = [line.strip() for line in f if line.strip()]
        if not accessions:
            logging.error(f"No accessions found in {args.accessions_file}.")
            sys.exit(1)
        logging.info(f"Loaded {len(accessions)} accessions to process.")
    except FileNotFoundError:
        logging.error(f"Accessions file not found: {args.accessions_file}")
        sys.exit(1)

    # --- 3. Create Task List ---
    # Create a list of tuples, where each tuple contains
    # the arguments for one call to process_accession
    tasks = [(acc, args) for acc in accessions]

    # --- 4. Run in Parallel ---
    logging.info("Starting parallel processing pool...")
    start_time = time.time()

    with Pool(processes=args.jobs) as pool:
        # pool.map runs the function on each item in 'tasks'
        # and blocks until all are complete.
        results = pool.map(process_accession, tasks)

    end_time = time.time()

    # --- 5. Report Results ---
    success_count = sum(1 for r in results if r)
    fail_count = len(results) - success_count

    logging.info("--- Run Complete ---")
    logging.info(f"Total wall clock time: {end_time - start_time:.2f} seconds.")
    logging.info(f"Successfully processed: {success_count} accessions.")
    logging.info(f"Failed to process: {fail_count} accessions.")

    if fail_count > 0:
        logging.warning("Some jobs failed. Check log above for details.")
        sys.exit(1)  # Exit with a non-zero code to indicate failure
    else:
        logging.info("All jobs completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
