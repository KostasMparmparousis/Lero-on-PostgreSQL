import argparse

from utils import *
import os
import socket
from config import *
from multiprocessing import Pool
import random
import glob
import pickle
import time
import psycopg2
import json
import sys
import shutil
from pathlib import Path
import subprocess

class PolicyEntity:
    def __init__(self, score) -> None:
        self.score = score

    def get_score(self):
        return self.score


class CardinalityGuidedEntity(PolicyEntity):
    def __init__(self, score, card_str) -> None:
        super().__init__(score)
        self.card_str = card_str


class PgHelper():
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file

    def start(self, pool_num):
        pool = Pool(pool_num)
        for fp, q in self.queries:
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, True, None, None))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


class LeroHelper():
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file, 
                test_queries, model_prefix, topK, workload_dir) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)

        # Create checkpoint directory
        self.checkpoint_dir = Path("../../../../models/experiment5/5.1/complexity_generalization/LERO/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.all_checkpoints = []

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def start(self, pool_num):
        REPEAT_COUNT = 5
        query_sequence = self.queries * REPEAT_COUNT
        lero_chunks = list(self.chunks(query_sequence, self.query_num_per_chunk))

        run_args = self.get_run_args()
        for c_idx, chunk in enumerate(lero_chunks):
            pool = Pool(pool_num)
            for fp, q in chunk:
                self.run_pairwise(q, fp, run_args, self.output_query_latency_file, self.output_query_latency_file + "_exploratory", pool)
            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()

            model_name = self.model_prefix + "_" + str(c_idx)
            self.retrain(model_name)
            self.test_benchmark(self.output_query_latency_file + "_" + model_name)

    def retrain(self, model_name):
        # Save checkpoint before retraining
        checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        training_data_file = self.output_query_latency_file + ".training"
        create_training_file(training_data_file, self.output_query_latency_file, self.output_query_latency_file + "_exploratory")
        print("retrain Lero model:", model_name, "with file", training_data_file)

        cmd = [
            "python3.8", "train.py",
            "--training_data", os.path.abspath(training_data_file),
            "--model_name", model_name,
            "--training_type", "1",
            "--checkpoint_path", os.path.abspath(checkpoint_path)
        ]
        print("run cmd:", " ".join(cmd))
        
        with open("train_stdout.log", "a") as out, open("train_stderr.log", "a") as err:
            subprocess.run(cmd, cwd=self.lero_server_path, stdout=out, stderr=err)

        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type":"load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        print(reply_json)
        os.system("sync")

    def test_benchmark(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
            do_run_query(q, fp, run_args, output_file, True, None, None)

    def get_run_args(self):
        run_args = []
        # LERO configuration
        run_args = [
            "SET enable_lero TO True",
            f"SET lero_server_host TO '{LERO_SERVER_HOST}'",
            f"SET lero_server_port TO {LERO_SERVER_PORT}"
        ]        
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args

    def write_card_file_via_udf(self, file_name, content):
        """Write cardinality file using PostgreSQL UDF"""
        try:
            print(f"Writing card file {file_name} via UDF...")
            print("content:", content)
            # Connect to PostgreSQL and execute the UDF
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT write_lero_card_file(%s, %s)",
                        (file_name, content)
                    )
                    success = cur.fetchone()[0]
                    if not success:
                        raise RuntimeError(f"UDF failed to write card file {file_name}")
        except Exception as e:
            print(f"Error writing card file via UDF: {str(e)}")
            raise

    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool):
        explain_query(q, run_args)
        policy_entities = []
        with open(self.lero_card_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split(";") for line in lines]
            for line in lines:
                policy_entities.append(CardinalityGuidedEntity(float(line[1]), line[0]))

        policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
        policy_entities = policy_entities[:self.topK]
        # exit(0)
        i = 0
        for entity in policy_entities:
            if isinstance(entity, CardinalityGuidedEntity):
                card_str = "\n".join(entity.card_str.strip().split(" "))
                # ensure that the cardinality file will not be changed during planning
                card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                #card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                card_file_path = os.path.join("cardinalities", card_file_name)
                with open(card_file_path, "w") as card_file:
                    card_file.write(card_str)

                # Use UDF to write the file inside the container
                self.write_card_file_via_udf(card_file_name, card_str)
                output_file = output_query_latency_file if i == 0 else exploratory_query_latency_file
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name), output_file, True, None, None))
                i += 1

    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type":"predict", "Plan":plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']

def load_queries_from_directory(directory_path, test_split=0.2, workload_order_file=None):
    """Load SQL queries from .sql files in a directory and split into train/test sets"""
    query_files = glob.glob(os.path.join(directory_path, "*.sql"))
    
    if workload_order_file:
        # Load ordered queries from the workload order file
        with open(workload_order_file, 'r') as f:
            ordered_files = [line.strip() for line in f if line.strip()]
        
        # Create full paths for ordered files
        train_queries = []
        for filename in ordered_files:
            full_path = os.path.join(directory_path, filename)
            if os.path.exists(full_path):
                with open(full_path, 'r') as qf:
                    query = qf.read().strip()
                    if query:
                        train_queries.append((filename, query))
            else:
                print(f"Warning: Query file {filename} from order file not found in directory")
        
        # All other queries become test queries
        remaining_files = set(os.path.basename(f) for f in query_files) - set(ordered_files)
        test_queries = []
        for filename in remaining_files:
            full_path = os.path.join(directory_path, filename)
            with open(full_path, 'r') as qf:
                query = qf.read().strip()
                if query:
                    test_queries.append((filename, query))
        
        print(f"Loaded {len(train_queries)} training queries (from order file) "
              f"and {len(test_queries)} test queries")
    else:
        # Original random split behavior
        queries = []
        for file_path in query_files:
            with open(file_path, 'r') as f:
                query = f.read().strip()
                if query:
                    file_name = os.path.basename(file_path)
                    queries.append((file_name, query))
        
        random.shuffle(queries)
        split_idx = int(len(queries) * (1 - test_split))
        train_queries = queries[:split_idx]
        test_queries = queries[split_idx:]
        print(f"Randomly split into {len(train_queries)} training queries "
              f"and {len(test_queries)} test queries")
    
    return train_queries, test_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_dir",
                        metavar="PATH",
                        help="Directory containing SQL query files")
    parser.add_argument("--test_split",
                        type=float,
                        default=0.0,
                        help="Fraction of queries to use for testing (default: 0.0)")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--query_num_per_chunk", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int)
    parser.add_argument("--topK", type=int)
    parser.add_argument("--workload_order_file",
                    metavar="PATH",
                    help="Text file specifying the order of training queries")    
    args = parser.parse_args()

    query_dir = args.query_dir
    test_split = args.test_split
    workload_order_file = args.workload_order_file
    print(f"Load queries from directory {query_dir}, test split = {test_split}")
    if workload_order_file:
        print(f"Using workload order from: {workload_order_file}")
    
    queries, test_queries = load_queries_from_directory(
        query_dir, 
        test_split,
        workload_order_file
    )
    print(f"Read {len(queries)} training queries and {len(test_queries)} test queries.")
    output_query_latency_file = args.output_query_latency_file
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 10
    if args.pool_num:
        pool_num = args.pool_num
    print("pool_num:", pool_num)

    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if args.algo:
        assert args.algo.lower() in ALGO_LIST
        algo = args.algo.lower()
    print("algo:", algo)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        print("Read", len(test_queries), "test queries.")

        query_num_per_chunk = args.query_num_per_chunk
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if args.model_prefix:
            model_prefix = args.model_prefix
        print("model_prefix:", model_prefix)

        topK = 5
        if args.topK is not None:
            topK = args.topK
        print("topK", topK)
        
        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK, query_dir)
        helper.start(pool_num)
