import argparse

from utils import *
import glob
import os
import random

def ensure_lero_directory(sql_file_path):
    """Ensure a LERO directory exists for the SQL file"""
    dir_path = os.path.dirname(sql_file_path)
    lero_dir = os.path.join(dir_path, "LERO")
    os.makedirs(lero_dir, exist_ok=True)
    return lero_dir

def save_plan(plan, output_path):
    """Save execution plan to a file"""
    with open(output_path, 'w') as f:
        json.dump(plan, f, indent=2)

def find_sql_files(directory):
    """Recursively find all .sql files in a directory"""
    sql_files = []
    file_paths = []
    queryIDs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.sql'):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
                queryIDs.append(file[:-4])  # Remove .sql extension for queryID
                sql_files.append(open(full_path, 'r').read())
    return [queryIDs, file_paths, sql_files]

NUM_EXECUTIONS = 3
# python test.py --query_path ../reproduce/test_query/stats.txt --output_query_latency_file stats.test
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--output_query_latency_file", metavar="PATH")

    args = parser.parse_args()
    test_queries = find_sql_files(args.query_path)
    for i in range(len(test_queries[0])):
        queryID = test_queries[0][i]
        fp = test_queries[1][i]
        q = test_queries[2][i]
        count = 0
        lero_dir = ensure_lero_directory(fp)
        while count < NUM_EXECUTIONS:
            query_plan = do_run_query(q, fp, ["SET enable_lero TO True", f"SET lero_server_host TO '{LERO_SERVER_HOST}'", f"SET lero_server_port TO {LERO_SERVER_PORT}"], args.output_query_latency_file, True, None, None)
            if query_plan is not None:
                output_path = os.path.join(lero_dir, f"run{count+1}" ,f"{queryID}_plan.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_plan(query_plan, output_path)
                print(f"Execution plan for {queryID} saved to {output_path}.")
            else:
                print(f"Failed to execute query {queryID}.")
            count += 1