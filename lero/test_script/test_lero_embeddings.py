import argparse

from utils import *
import glob
import os
import random
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(SCRIPT_DIR)
TEMP_EMBEDDING_FILE = os.path.join(SERVER_DIR, "lero_last_embedding.json")
NUM_EXECUTIONS = 1
from tqdm import tqdm
import socket
import time

def ensure_lero_directory(sql_file_path):
    """Ensure a LERO directory exists for the SQL file"""
    dir_path = os.path.dirname(sql_file_path)
    lero_dir = os.path.join(dir_path, "LERO")
    os.makedirs(lero_dir, exist_ok=True)
    return lero_dir

# --- HELPER FUNCTIONS ---
def save_json(data, output_path):
    try:
        with open(output_path, 'w') as f: json.dump(data, f, indent=2)
    except Exception as e: print(f"  - WARNING: Failed to save JSON to {output_path}: {e}")

def _load_json_safely(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: return json.load(f)
    except: return None

def find_sql_files(directory, skip_lero_processed=False):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.sql'):
                if skip_lero_processed and os.path.exists(os.path.join(root, "LERO")):
                    continue
                yield os.path.join(root, file), file

def get_embedding_from_server(plan_json):
    if not plan_json or 'Plan' not in plan_json[0]:
        print("  - ERROR: Invalid plan JSON provided for embedding generation.")
        return None
        
    if os.path.exists(TEMP_EMBEDDING_FILE): os.remove(TEMP_EMBEDDING_FILE)
    
    # The message body must match what the server's feature generator expects
    message_data = {"msg_type": "get_embedding_for_plan", "Plan": plan_json[0]['Plan']}
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
            sock.sendall(bytes(json.dumps(message_data) + "*LERO_END*", "utf-8"))
            sock.recv(1024)
    except Exception as e:
        print(f"  - ERROR: Could not connect to Lero server for PG embedding: {e}")
        return None
    
    # Robustly wait for and read the temp file
    embedding_data = None
    start_wait = time.time()
    while time.time() - start_wait < 5.0:
        embedding_data = _load_json_safely(TEMP_EMBEDDING_FILE)
        if embedding_data: break
        time.sleep(0.1)
    return embedding_data

def run_lero_phase(sql_query, file_path, extra_settings=[]):
    """A helper to run one phase of the Lero process and get plan + embedding."""
    if os.path.exists(TEMP_EMBEDDING_FILE):
        os.remove(TEMP_EMBEDDING_FILE)
    
    # Base settings that are always required
    base_settings = [
        "SET enable_lero TO True",
    ]
    
    # This call runs the query and triggers the server-side hook
    plan_json = test_query(sql_query, file_path, base_settings + extra_settings)
    
    # Robustly wait for and read the embedding file
    embedding_data = None
    wait_timeout = 5.0
    start_wait = time.time()
    while time.time() - start_wait < wait_timeout:
        embedding_data = _load_json_safely(TEMP_EMBEDDING_FILE)
        if embedding_data:
            break
        time.sleep(0.1)
        
    return plan_json, embedding_data

# --- MAIN EXECUTION LOGIC ---
def generate_analysis_artifacts(workload_dir, skip_processed=False):
    sql_files_generator = find_sql_files(workload_dir, skip_lero_processed=skip_processed)
    
    for full_path, filename in tqdm(list(sql_files_generator), desc="Processing Queries"):
        query_id = filename.replace('.sql', '')
        lero_dir = os.path.join(os.path.dirname(full_path), "LERO")
        os.makedirs(lero_dir, exist_ok=True)
        
        with open(full_path, 'r') as f: sql_query = f.read()
            
        print(f"\nProcessing: {filename}")
        
        try:
            # --- 1. GET LERO PLAN AND EMBEDDING ---
            print("  - Getting Lero plan and embedding...")
            if os.path.exists(TEMP_EMBEDDING_FILE): os.remove(TEMP_EMBEDDING_FILE)
            
            lero_plan_json = test_query(sql_query, full_path, ["SET enable_lero TO True"], 'stats.test')
            time.sleep(0.5) # Wait for the last of many embeddings to be written
            lero_embedding_data = _load_json_safely(TEMP_EMBEDDING_FILE)
            
            save_json(lero_plan_json, os.path.join(lero_dir, f"{query_id}_lero_plan.json"))
            save_json(lero_embedding_data, os.path.join(lero_dir, f"{query_id}_lero_embedding.json")) if lero_embedding_data else print("  - WARNING: No Lero embedding found.")
            print("  - Lero files saved.")

            # --- 2. GET POSTGRESQL PLAN AND EMBEDDING ---
            print("  - Getting PostgreSQL plan and embedding...")
            pg_plan_json = test_query(sql_query, full_path, ["SET enable_lero TO False"], 'stats.test')
            
            pg_embedding_data = get_embedding_from_server(pg_plan_json)
            
            save_json(pg_plan_json, os.path.join(lero_dir, f"{query_id}_postgres_plan.json"))
            save_json(pg_embedding_data, os.path.join(lero_dir, f"{query_id}_postgres_embedding.json")) if pg_embedding_data else print("  - WARNING: No PostgreSQL embedding found.")
            print("  - PostgreSQL files saved.")
            
            clear_cache()

        except Exception as e:
            print(f"  - FATAL ERROR processing {filename}: {e}")
            clear_cache()
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Lero artifact generation script")
    parser.add_argument("workload_path", help="Path to the directory with SQL queries")
    parser.add_argument("--skip_lero_processed", action="store_true", help="Skip queries that already have a LERO directory")
    args = parser.parse_args()
    
    # It's assumed the Lero server is already running.
    generate_analysis_artifacts(args.workload_path, skip_processed=args.skip_lero_processed)
    
    print("\nProcessing complete.")