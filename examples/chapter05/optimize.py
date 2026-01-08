import datetime
import os
import re
import subprocess
from typing import Any, Dict

import optuna
from optuna.trial import Trial


def assign_params_to_file(
    input_filepath: str,
    output_filepath: str,
    params: Dict[str, Any],
) -> None:
    """
    Assign parameters in a file with given values.

    Args:
        input_filepath (str): Path to the input file.
        output_filepath (str): Path to the output file.
        params (Dict[str, Any]): Dictionary of parameters to assign.
    """
    with open(input_filepath, "r") as f:
        content = f.read()
    for param_name, value in params.items():
        content = content.replace(f"${{{param_name}}}", f"{value}")
    with open(output_filepath, "w") as f:
        f.write(content)


def objective(trial: Trial, study_dir):
    global_batch_size = 32
    num_gpus = 4
    micro_batch_size = trial.suggest_categorical(
        "micro_batch_size", [1, 2, 4, 8, 16, 32]
    )
    zero_stage = trial.suggest_int("zero_stage", 0, 3)
    if zero_stage == 3:
        autotp_size = 1
    else:
        autotp_size = trial.suggest_categorical("autotp_size", [1, 2, 4])

    # mbs * (num_gpus // tp) * gradient_accumulation_steps == global_batch_size
    if (
        global_batch_size % (micro_batch_size * (num_gpus // autotp_size))
        != 0
    ):
        raise optuna.TrialPruned()
    gradient_accumulation_steps = global_batch_size // (
        micro_batch_size * (num_gpus // autotp_size)
    )

    trial_id = trial._trial_id
    train_script_template = "run_optimize_template.sh"
    train_script = os.path.join(study_dir, f"run_{trial_id}.sh")
    deepspeed_config_template = "configs/ds_config_optimize_template.json"
    deepspeed_config = os.path.join(study_dir, f"ds_config_{trial_id}.json")
    assign_params_to_file(
        train_script_template,
        train_script,
        {
            "per_device_train_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "deepspeed_config_path": deepspeed_config,
        },
    )
    assign_params_to_file(
        deepspeed_config_template,
        deepspeed_config,
        {
            "zero_stage": zero_stage,
            "autotp_size": autotp_size,
            "allgather_partitions": trial.suggest_categorical(
                "allgather_partitions", ["true", "false"]
            ),
            "overlap_comm": trial.suggest_categorical(
                "overlap_comm", ["true", "false"]
            ),
            "reduce_scatter": trial.suggest_categorical(
                "reduce_scatter", ["true", "false"]
            ),
            "contiguous_gradients": trial.suggest_categorical(
                "contiguous_gradients", ["true", "false"]
            ),
        },
    )

    log_file = os.path.join(study_dir, f"trial_{trial_id}.log")
    command = f"bash {train_script} > {log_file} 2>&1"
    subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with open(log_file, "r") as f:
        log = f.read()

    try:
        ret = re.findall("'train_runtime': (.*?),", log)
        train_runtime = float(ret[0])
        return train_runtime
    except Exception:
        raise optuna.TrialPruned()


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = f"study_{timestamp}"
    study_dir = os.path.join("optimize", study_name)
    os.makedirs(study_dir, exist_ok=True)
    study = optuna.create_study(
        storage="sqlite:///" + os.path.join(study_dir, "study.db"),
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, study_dir),
        n_trials=10,
    )


if __name__ == "__main__":
    main()
