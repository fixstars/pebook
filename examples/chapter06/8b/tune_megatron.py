import argparse
import os
import re
import statistics
from typing import Dict, Optional, Union
import math

import optunahub
import optuna
from optuna.trial import Trial
from aibooster.intelligence.zenith_tune import CommandOutputTuner
from aibooster.intelligence.zenith_tune.utils import replace_params_to_file

# NUM_LAYERS=80
NUM_LAYERS=32
NUM_QUERY_GROUPS=8
GLOBAL_BATCH_SIZE=16

TARGET_SCRIPT = "train_template.sh"

# 1ノードの場合
NUM_GPUS=8
PARAMETER_DISTRIBUTIONS = {
    "micro_batch_size": optuna.distributions.IntDistribution(1, 16),
    "tensor_model_parallel_size": optuna.distributions.IntDistribution(1, 8),
    "pipeline_model_parallel_size": optuna.distributions.IntDistribution(1, 8),
    "context_model_parallel_size": optuna.distributions.IntDistribution(1, 8),
    # "fp": optuna.distributions.CategoricalDistribution(["fp8", "fp16"]),
    # "recompute_granularity": optuna.distributions.CategoricalDistribution(["None", "selective", "full"]),
    # "recompute_method": optuna.distributions.CategoricalDistribution(["None", "block", "uniform"]),
    "recompute_num_layers": optuna.distributions.IntDistribution(0, NUM_LAYERS),
    "empty_unused_memory_level": optuna.distributions.IntDistribution(0, 2),
}

# 約数計算
def get_divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))
DIVISORS_GPUS=get_divisors(NUM_GPUS)
DIVISORS_QUERY=get_divisors(NUM_QUERY_GROUPS)
DIVISORS_LAYER=get_divisors(NUM_LAYERS)


# TPの候補を指定GPU数の約数かつQUERYグループ数の約数として指定
TP_CANDIDATES = [
    d for d in sorted(list(set(DIVISORS_GPUS) & set(DIVISORS_QUERY)))
    if PARAMETER_DISTRIBUTIONS["tensor_model_parallel_size"].low <= d <= PARAMETER_DISTRIBUTIONS["tensor_model_parallel_size"].high
]

# PPの候補はGPU数/指定されたTPの約数
def get_pp_candidates(tensor_model_parallel_size):
    divsors_gpus_div_tp = get_divisors(NUM_GPUS // tensor_model_parallel_size)
    return [
        d for d in sorted(list(set(DIVISORS_LAYER) & set(divsors_gpus_div_tp)))
        if PARAMETER_DISTRIBUTIONS["pipeline_model_parallel_size"].low <= d <= PARAMETER_DISTRIBUTIONS["pipeline_model_parallel_size"].high
    ]

# CPの候補はGPU数/(指定されたTP * 指定されたPP)の約数
def get_cp_candidates(tensor_model_parallel_size, pipline_model_parallel_size):
    divsors_gpus_div_tp_pp = get_divisors(NUM_GPUS // (tensor_model_parallel_size * pipline_model_parallel_size))
    return [
        d for d in sorted(list(set(DIVISORS_LAYER) & set(divsors_gpus_div_tp_pp)))
        if PARAMETER_DISTRIBUTIONS["context_model_parallel_size"].low <= d <= PARAMETER_DISTRIBUTIONS["context_model_parallel_size"].high
    ]

def get_micro_batch_size_candidates(maximum_micro_batches):
    divsors_maximum_micro_batches = get_divisors(maximum_micro_batches)
    return [
        d for d in sorted(divsors_maximum_micro_batches)
        if PARAMETER_DISTRIBUTIONS["micro_batch_size"].low <= d <= PARAMETER_DISTRIBUTIONS["micro_batch_size"].high
    ]

def get_recompute_num_layers_candidates(pipeline_model_parallel_size):
    layer_per_stage = PARAMETER_DISTRIBUTIONS["recompute_num_layers"].high // pipeline_model_parallel_size
    return [0] + [
        d for d in get_divisors(layer_per_stage)
        if PARAMETER_DISTRIBUTIONS["recompute_num_layers"].low <= d <=  PARAMETER_DISTRIBUTIONS["recompute_num_layers"].high
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--study-name")
    parser.add_argument(
        "--n-trials",
        default=10,
        help="The number of optimization steps.",
        type=int,
    )
    parser.add_argument(
        "--use-ingo",
        action="store_true",
        help="Use ingo sampler for optimization.",
    )
    return parser.parse_args()


def command_generator(
    trial: Trial,
    trial_id: int,
    study_dir: str,
    dist_info: Dict[str, Union[int, str]],
    **kwargs,
) -> str:
    global_batch_size = GLOBAL_BATCH_SIZE
    tp_index = trial.suggest_int("tensor_model_parallel_size_index", 0, len(TP_CANDIDATES) - 1)
    tensor_model_parallel_size = TP_CANDIDATES[tp_index]
    trial.set_user_attr("tensor_model_parallel_size", tensor_model_parallel_size)

    pp_candidates = get_pp_candidates(tensor_model_parallel_size)
    pp_index = trial.suggest_int("pipeline_model_parallel_size_index", 0, len(pp_candidates) - 1)
    pipeline_model_parallel_size = pp_candidates[pp_index]
    trial.set_user_attr("pipeline_model_parallel_size", pipeline_model_parallel_size)

    cp_candidates = get_cp_candidates(tensor_model_parallel_size, pipeline_model_parallel_size)
    cp_index = trial.suggest_int("context_model_parallel_size_index", 0, len(cp_candidates) - 1)
    context_model_parallel_size = cp_candidates[cp_index]
    trial.set_user_attr("context_model_parallel_size", context_model_parallel_size)

    world_size = NUM_GPUS

    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size)

    if global_batch_size % data_parallel_size != 0:
        print("ValueError: global_batch_size % data_parallel_size  =", global_batch_size % data_parallel_size )
        return None
    maximum_micro_batches = global_batch_size // data_parallel_size
    micro_batch_candidates = get_micro_batch_size_candidates(maximum_micro_batches)
    micro_batch_index = trial.suggest_int("micro_batch_size_index", 0, len(micro_batch_candidates) - 1)
    micro_batch_size = micro_batch_candidates[micro_batch_index]
    trial.set_user_attr("micro_batch_size", micro_batch_size)

    recompute_num_layers_candidates = get_recompute_num_layers_candidates(pipeline_model_parallel_size)
    recompute_num_layers_index = trial.suggest_int("recompute_num_layers_index", 0, len(recompute_num_layers_candidates) - 1)
    recompute_num_layers = recompute_num_layers_candidates[recompute_num_layers_index]
    trial.set_user_attr("recompute_num_layers", recompute_num_layers)

    empty_unused_memory_level = trial.suggest_int("empty_unused_memory_level", low=PARAMETER_DISTRIBUTIONS["empty_unused_memory_level"].low, high=PARAMETER_DISTRIBUTIONS["empty_unused_memory_level"].high)

    tuning_script_path = os.path.join(study_dir, f"train_{trial_id}.sh")

    if dist_info["rank"] == 0:
        replace_params_to_file(
            TARGET_SCRIPT,
            tuning_script_path,
            {
                "global_batch_size": global_batch_size,
                "micro_batch_size": micro_batch_size,
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "pipeline_model_parallel_size": pipeline_model_parallel_size,
                "context_model_parallel_size": context_model_parallel_size,
                "recompute_num_layers": recompute_num_layers,
                "empty_unused_memory_level": empty_unused_memory_level,
                # "fp": fp,
                # "recompute_granularity": optuna.distributions.CategoricalDistribution(["None", "selective", "full"]),
                # "recompute_method": optuna.distributions.CategoricalDistribution(["None", "block", "uniform"]),
            },
        )

    command = f"bash {tuning_script_path}"
    return command


def value_extractor(log_path: str) -> Optional[float]:
    with open(log_path) as f:
        lines = f.readlines()

    sample_flops = []
    for line in lines:
        if "throughput per GPU" in line:
            match = re.findall(
                r"iteration\s+(\d+)/\s+\d+.*throughput per GPU \(TFLOP/s/GPU\): (\d+\.\d+)",
                line,
            )[0]
            iteration = int(match[0])
            tflops_per_gpu = float(match[1])

            # 10iterのうち最初のサンプルを除くiterのTFLOP/s/GPUを抽出
            if 2 <= iteration and iteration <= 10:
                sample_flops.append(tflops_per_gpu)
    # 指定のiter数に満たなければ、タイムアウトなどのエラーが発生したと見なしNoneをreturn
    if len(sample_flops) < 9:
        return None
    return statistics.harmonic_mean(sample_flops)


def main():
    args = parse_args()
    print(args)
    sampler = optuna.samplers.TPESampler()

    tuner = CommandOutputTuner(
        args.output_dir, args.study_name, sampler=sampler, maximize=True)
    tuner.optimize(command_generator, value_extractor, args.n_trials,
        default_params={
            'tensor_model_parallel_size_index': 0, 'pipeline_model_parallel_size_index': 0,
            'context_model_parallel_size_index': 2, 'micro_batch_size_index': 1,
            'recompute_num_layers_index': 1, 'empty_unused_memory_level': 2
        },
    )
    tuner.analyze()


if __name__ == "__main__":
    main()
