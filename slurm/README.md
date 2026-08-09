# Slurm Job Submission for itunad

## TL;DR

If `ituna` with `trigger_type='manual'` tells you to run a command like this:

```bash
ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir ~/git/ituna/.cache
```

You can run the following command instead to submit the same job to a Slurm cluster. Just prepend `sbatch slurm/SCRIPT_NAME.sh` to the command:

**KIT:**

```bash
sbatch slurm/KIT_run.sh ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir ~/git/ituna/.cache
```

**JUWELS:**

```bash
sbatch slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir ~/git/ituna/.cache
```

Alternatively, with more compute ressources:

**KIT: 4 GPUs, 20 tasks**

```bash
sbatch --gres=gpu:full:4 --ntasks=20 slurm/KIT_run.sh ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir ~/git/ituna/.cache
```

**JUWELS: 4 GPUs, 20 tasks (8 slurm jobs)**

```bash
sbatch --array=1-8 --ntasks=20 slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir ~/git/ituna/.cache
```

---

This directory contains scripts for submitting training jobs to different Slurm-managed clusters.

## Prerequisites

Before running the submission scripts, please ensure your environment is set up correctly.

### 1. Conda Installation

The submission scripts rely on a Conda installation to activate the correct Python environment.

- The path to your Conda installation should be specified via the `CONDA_BASE_PATH` environment variable.
- If this variable is not set, it will default to `$HOME/miniconda3`.
- The name of the conda environment can be set with `CONDA_ENV_NAME`, which defaults to `ituna`.

### 2. Environment Variables

The `ituna` library, particularly the DataJoint backend, requires database credentials and other configuration to be set as environment variables.

It is recommended to create a `.env` file in the root of this repository to manage these variables. You can create this file by copying a template and filling in your specific values.

**Example `.env` file:**

```bash
# DataJoint database credentials
export DJ_HOST="your-database-host"
export DJ_USER="your-username"
export DJ_PASS="your-password"

# Path to Conda installation
export CONDA_BASE_PATH="/path/to/your/miniconda3"

# Name of the Conda environment
export CONDA_ENV_NAME="ituna"
```

**Important:** Remember to source this file (e.g., `source .env`) in your shell before submitting a Slurm job, or add the source command to your `.bashrc` or `.zshrc` file to have the variables loaded automatically.

## General Usage

The submission scripts are wrappers that execute any command you pass to them on the Slurm cluster. To use them, prepend `sbatch slurm/SCRIPT_NAME.sh` to the command you want to run.

For example, to run the `ituna-fit-distributed-datajoint` command:

```bash
sbatch <path-to-script> ituna-fit-distributed-datajoint --sweep-name hippocampus-cebra-time-grid --schema-name ituna_cebra_v1 --cache-dir /path/to/your/.cache
```

See `ituna-fit-distributed-datajoint --help` for all available arguments.

## Cluster-specific Scripts

### KIT (`KIT_run.sh`)

This script is configured for the KIT cluster.

#### Basic Example

To run a sweep, prepend `sbatch slurm/KIT_run.sh` to your command:

```bash
sbatch slurm/KIT_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

This will use the default SLURM settings in the script (`--gres=gpu:full:1`).

#### Running Large Batch Jobs

You can leverage Slurm to run many training jobs in parallel across multiple GPUs. This is controlled by the `--gres` and `--ntasks` options of `sbatch`.

- `--gres=gpu:full:X`: This requests `X` GPUs for your job. On the KIT cluster, `X` can be 1, 2, 3, or 4.
- `--ntasks=Y`: This tells Slurm to run `Y` tasks (processes) for your job.

The `KIT_run.sh` script will automatically distribute the `Y` tasks among the `X` available GPUs in a round-robin fashion. Each task will run the `ituna-fit-distributed-datajoint` command with the same arguments you provided.

**Example:**

Suppose you want to run 16 training jobs in parallel, distributed across 4 GPUs. You can submit the job as follows:

```bash
sbatch --gres=gpu:full:4 --ntasks=16 slurm/KIT_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

In this case:

- Slurm allocates 4 GPUs to your job.
- Slurm starts 16 parallel tasks.
- The script assigns each task to one of the 4 GPUs. Each GPU will be shared by 4 tasks (`16 tasks / 4 GPUs = 4 tasks per GPU`). The script manages this by setting the `CUDA_VISIBLE_DEVICES` environment variable for each task.

This setup is ideal for hyperparameter sweeps where each job is independent. The `ituna-fit-distributed-datajoint` script with the `--order random` argument (the default) will ensure that each of the 16 parallel processes picks a different, random, unprocessed model from the sweep to train.

### JUWELS (`JUWELS_run.sh`)

This script is configured for the JUWELS cluster.

#### Basic Example

The script has default values for a multi-GPU job (`--nodes=1`, which on JUWELS provides 4 GPUs, and `--ntasks=4`).

```bash
sbatch slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

You can override the number of tasks when submitting:

```bash
sbatch --ntasks=1 slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

#### Slurm Configuration Notes

- **Account:** You may need to edit the script to specify your account with `#SBATCH --account=<your-account>`. The script defaults to `hai_mechanistic`.
- **GPU Resources**: All nodes on the `booster` and `develbooster` partitions are equipped with 4 A100 GPUs. You request resources per node (`--nodes=X`).
- **Partitions:**
    - `booster`: For regular jobs. It's recommended to specify a walltime with `--time=HH:MM:SS` to get your jobs started more quickly.
    - `develbooster`: For short, development jobs. This partition provides faster access but requires you to specify `--time=02:00:00` or less.
- **Time Limit**: It is always a good practice to specify a time limit for your job with `--time`.

#### Running Large Batch Jobs

On JUWELS, `srun` automatically handles the distribution of tasks across the allocated GPUs. You do not need the manual `CUDA_VISIBLE_DEVICES` assignment that is used in the KIT script. Slurm's process management will ensure that tasks are spread across the available GPUs.

For example, if you request 1 node (which has 4 GPUs) and 20 tasks:

```bash
sbatch --ntasks=20 slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

Slurm will schedule the 20 `ituna-fit-distributed-datajoint` processes across the 4 GPUs, with each GPU handling 5 processes.

##### Submitting Many Jobs with Job Arrays

For very large sweeps, you can use Slurm's job array feature to submit multiple jobs with a single command.

A large submission could look like this:

```bash
sbatch --array=1-16 --ntasks=20 --time=08:00:00 slurm/JUWELS_run.sh ituna-fit-distributed-datajoint --sweep-name <your-sweep-name>
```

This command submits a job array with 16 jobs. Each of these 16 jobs will:

- Run for a maximum of 8 hours.
- Request one node with 4 GPUs.
- Run 20 model training tasks in parallel, distributed across the 4 GPUs (5 tasks per GPU).

In total, this would run `16 jobs * 20 models/job = 320` model trainings in parallel, assuming all jobs receive resources at the same time.
