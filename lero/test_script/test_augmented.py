import argparse

from utils import *
import glob
import os
import random
from utils import clear_cache
import os
import json

def save_metrics_from_plan(query_name, sql_text, plan_json, predicted_latency, output_path):
    """
    Save query metrics (planning time, actual latency, predicted latency) to JSON.
    
    Args:
        query_name (str): Identifier of the query.
        sql_text (str): Full SQL query text.
        plan_json (list): Execution plan JSON returned by PostgreSQL.
        predicted_latency (float): Predicted latency from the model.
        output_path (str): Path to save the metrics JSON.
    """
    # Extract planning time and actual total execution time
    if plan_json and isinstance(plan_json, list) and "Planning Time" in plan_json[0]:
        planning_time = plan_json[0]["Planning Time"]
        actual_latency = plan_json[0]["Execution Time"]
    else:
        print(f"WARNING: Could not find planning/execution times in plan for {query_name}.")
        planning_time = None
        actual_latency = None

    metrics = {
        "query_name": query_name,
        "sql": sql_text,
        "planning_time": planning_time,
        "predicted_latency": predicted_latency,
        "actual_latency": actual_latency
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")
    except Exception as e:
        print(f"ERROR saving metrics for {query_name}: {e}")


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

def find_sql_files(directory, skip_lero_processed=False):
    """Recursively find all .sql files in a directory"""
    sql_files = []
    file_paths = []
    queryIDs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.sql'):
                full_path = os.path.join(root, file)
                
                if skip_lero_processed:
                    lero_dir = os.path.join(os.path.dirname(full_path), "LERO")
                    if os.path.exists(lero_dir):
                        print(f"Skipping {file} as LERO directory already exists.")
                        continue

                file_paths.append(full_path)
                queryIDs.append(file[:-4])  # Remove .sql extension for queryID
                with open(full_path, 'r') as f:
                    sql_files.append(f.read())
    return [queryIDs, file_paths, sql_files]

import socket

def get_predicted_latency_from_server(plan_json):
    """Send plan to Lero server and get predicted latency."""
    if not plan_json or 'Plan' not in plan_json[0]:
        print("  - ERROR: Invalid plan JSON provided for latency prediction.")
        return None

    message_data = {"msg_type": "predict", "Plan": plan_json[0]['Plan']}

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
            sock.sendall(bytes(json.dumps(message_data) + "*LERO_END*", "utf-8"))
            response = sock.recv(8192).decode("utf-8")
            reply = json.loads(response)
            latency = reply.get("latency")
    except Exception as e:
        print(f"  - ERROR: Could not connect to Lero server for latency prediction: {e}")
        return None

NUM_EXECUTIONS = 1
# python test.py --query_path ../reproduce/test_query/stats.txt --output_query_latency_file stats.test
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--skip_lero_processed", action="store_true", help="Skip queries that already have a LERO directory")

    args = parser.parse_args()
    test_queries = find_sql_files(args.query_path, skip_lero_processed=args.skip_lero_processed)
    # --- The main loop that processes queries ---
    for i in tqdm(range(len(test_queries[0])), desc="Processing queries"):
        queryID = test_queries[0][i]
        fp = test_queries[1][i]
        q = test_queries[2][i]
        count = 0
        lero_dir = ensure_lero_directory(fp)
        # Loop for multiple executions per query
        while count < NUM_EXECUTIONS:
            print(f"\nExecuting {queryID} (Run {count + 1}/{NUM_EXECUTIONS})")
            query_plan = None
            try:
                # Execute the query
                query_plan = test_query(q, fp, ["SET enable_lero TO True"], args.output_query_latency_file, True, None, None)
            except Exception as e:
                print(f"ERROR executing query {queryID}: {e}")
                print("Attempting to restart database and retry after a delay...")
                # If execution fails, still try to restart the DB to recover
                clear_cache()
                continue # Skip to next execution attempt

            if query_plan is not None:
                # Save the successful plan
                if NUM_EXECUTIONS > 1:
                    output_path = os.path.join(lero_dir, f"run{count+1}", f"{queryID}_lero_plan.json")
                else:
                    output_path = os.path.join(lero_dir, f"{queryID}_lero_plan.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_plan(query_plan, output_path)
                print(f"Execution plan for {queryID} saved to {output_path}.")

                # Get predicted latency from Lero server
                predicted_latency = get_predicted_latency_from_server(query_plan)
                if predicted_latency is not None:
                    print(f"Predicted latency for {queryID}: {predicted_latency} ms")
                    if NUM_EXECUTIONS > 1:
                        metrics_path = os.path.join(lero_dir, f"run{count+1}", f"{queryID}_lero_metrics.json")
                    else:
                        metrics_path = os.path.join(lero_dir, f"{queryID}_lero_metrics.json")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_metrics_from_plan(
                        query_name=queryID,
                        sql_text=q,
                        plan_json=query_plan,
                        predicted_latency=predicted_latency,
                        output_path=metrics_path
                    )
            else:
                print(f"Failed to get an execution plan for query {queryID}.")
            count += 1