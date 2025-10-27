# Lero Optimizer

This directory contains the implementation of Lero, a learned query optimizer that follows a learning-to-rank paradigm to select the best query plan.

This guide provides instructions for using Lero within the evaluation suite.

### Prerequisites

1.  The main Docker environment for the evaluation suite must be built and running. It includes the version of PostgreSQL specifically patched for Lero.
2.  You have a Conda installation (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

---

## 1. Environment Setup

All Lero commands must be run from a dedicated Conda environment.

1.  **Navigate to the Lero directory:**
    ```bash
    cd optimizers/Lero-on-PostgreSQL/lero
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate lero
    ```

---

## 2. Running Lero

Lero uses a server-client model. You must start the Lero server in one terminal, then run training or testing scripts from a second terminal.

### Step 0: Configure Scripts
Before running, you must configure two files:
1.  **`lero/server.conf`:** This file configures the server, including the `ModelPath` for **testing/inference**.
2.  **`lero/test_script/config.py`:** This file configures the client scripts, including the database connection details. **Ensure this is set correctly before proceeding.**

### General Training Command
Training involves starting the server and then running the `train_model.py` script, which controls the training loop.

1.  **Start the Lero Server:** In your first terminal:
    ```bash
    # From the lero/ directory
    python3 server.py
    ```
2.  **Configure Training Script:** In your second terminal, **edit `lero/test_script/train_model.py`**. Modify the `self.checkpoint_dir` variable on line 63 to your desired output directory for saving checkpoints.
3.  **Run Training:** In the second terminal:
    ```bash
    cd test_script
    python3 train_model.py --query_dir <path/to/queries/> --output_query_latency_file <log_file.log> --model_prefix <model_name> [other_args...]
    ```
    *   `--query_dir`: Path to the training workload.
    *   `--output_query_latency_file`: The log file for executed plan latencies.
    *   `--model_prefix`: A prefix for the saved model checkpoint files.

### General Testing Command
Testing involves configuring the server with a pre-trained model path, starting it, and then running the `test.py` script.

1.  **Configure Server:** **Edit the `lero/server.conf` file**. Set the `ModelPath` variable on line 8 to the path of your pre-trained model.
2.  **Start the Lero Server:** In your first terminal:
    ```bash
    # From the lero/ directory
    python3 server.py
    ```
3.  **Run Testing:** In a second terminal:
    ```bash
    cd test_script
    python3 test.py --query_path <path/to/test/queries/> --output_query_latency_file <results_file.log>
    ```
    *   `--query_path`: Path to the directory containing the test queries.
    *   `--output_query_latency_file`: The file where test results will be logged.

---

## 3. Replicating Paper Experiments

For the exact commands, model paths, and setup needed to generate the results for each experiment (E1-E5) in our paper, refer to the detailed guide below.

👉 [**Lero Experiment Reproduction Commands**](experiments.md)

---

## 4. Reference from original Lero Documentation

<details>
<summary><b>Click to expand for key concepts from the original Lero documentation.</b></summary>

### Learning-to-Rank Paradigm

The core idea of Lero is that learning the relative order (or rank) of query plans is an easier and more robust machine learning task than predicting the absolute latency of each plan. By focusing on ranking, Lero aims to build a more effective learned optimizer.

### Modified PostgreSQL

Lero requires a modified version of PostgreSQL to communicate with its external server. The `Dockerfile` in this evaluation suite **already handles this for you**. It applies the necessary patch (`0001-init-lero.patch`) during the build process, so you do not need to manually download, patch, or compile PostgreSQL.

For more details on the original setup and architecture, please refer to the complete [`original_documentation.md`](original_documentation.md) file.

</details>