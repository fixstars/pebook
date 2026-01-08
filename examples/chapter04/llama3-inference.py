import argparse
import contextlib
import json
import math
import os
import random
import string
import time
from dataclasses import asdict, dataclass

import torch
import torch.profiler
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration & Argument Parsing ---


@dataclass
class BenchmarkConfig:
    """Configuration class for the LLM inference benchmark."""

    # Model and Dataset
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    draft_name: str = None
    dataset_name: str = "llm-jp/Synthetic-JP-EN-Coding-Dataset"

    # Inference Parameters
    device_map: str = "auto"
    num_samples: int = 128
    batch_size: int = 1
    max_new_tokens: int = 1000
    random_prompt_length: int = 0
    simple_prompt: str = None
    num_assistant_tokens: int = 20

    # Optimization Flags
    use_quantize: bool = False
    use_int8: bool = False
    use_flash_attention_2: bool = False
    use_speculative_decoding: bool = False
    use_large_speculative_decoding: bool = False

    # Profiler Settings
    use_profiler: bool = False
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    with_flops: bool = False
    with_modules: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_samples < self.batch_size:
            raise ValueError("Number of samples must be at least the batch size.")
        if self.use_int8:
            self.use_quantize = True  # INT8 is a form of quantization
        if self.use_speculative_decoding and (self.draft_name is None):
            raise ValueError(
                "Draft model name must be specified when using speculative decoding."
            )
        if self.random_prompt_length > 0 and self.simple_prompt:
            raise ValueError(
                "Cannot use both random prompt length and simple prompt simultaneously."
            )


def parse_args() -> BenchmarkConfig:
    """Parses command-line arguments and returns a BenchmarkConfig instance."""

    def str2bool(v):
        """Helper function to convert string to boolean."""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="LLM inference benchmark script.")

    # Dynamically add arguments from the BenchmarkConfig dataclass
    for field in BenchmarkConfig.__dataclass_fields__.values():
        arg_name = f"--{field.name.replace('_', '-')}"
        if field.type is bool:
            parser.add_argument(
                arg_name,
                action="store_true",
                default=field.default,
                help=f"Enable {field.name}. (default: {field.default})",
            )
        else:
            parser.add_argument(
                arg_name,
                type=field.type,
                default=field.default,
                help=f"{field.metadata.get('help', '')} (default: {field.default})",
            )

    args = parser.parse_args()
    return BenchmarkConfig(**vars(args))


def print_config(config: BenchmarkConfig):
    """Prints the benchmark configuration in a readable format."""
    print("=" * 60)
    print("🚀 Benchmarking Llama 3 Inference")
    print("▶ Model:")
    print(f"  - Main model: {config.model_name}")
    print(f"  - Draft model: {config.draft_name}")
    print("-" * 60)
    print("▶ General Settings:")
    print(f"  - Number of samples: {config.num_samples}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max new tokens: {config.max_new_tokens}")
    print(f"  - Num assistant tokens: {config.num_assistant_tokens}")
    print(f"  - Random prompt length: {config.random_prompt_length}")
    if config.random_prompt_length > 0:
        print("▶ Prompt Source: Random Prompt Mode ✅")
    elif config.simple_prompt:
        print("▶ Prompt Source: Simple Prompt Mode ✅")
        print(f"  - Prompt: '{config.simple_prompt}'")
    else:
        print(f"▶ Prompt Source: Dataset Mode ({config.dataset_name})")
    print("-" * 60)
    print("▶ Feature Flags:")
    print(f"  - Device Map: {config.device_map}")
    print(f"  - Use Quantize: {'✅' if config.use_quantize else '❌'}")
    if config.use_quantize:
        print(
            f"    - Use INT8 Quantization: {'✅' if config.use_int8 else '❌ (using NF4)'}"
        )
    print(
        f"  - Use Flash Attention 2: {'✅' if config.use_flash_attention_2 else '❌'}"
    )
    print(
        f"  - Use Speculative Decoding: {'✅' if config.use_speculative_decoding else '❌'}"
    )
    print(
        f"  - Use Large Speculative Decoding: {'✅' if config.use_large_speculative_decoding else '❌'}"
    )
    print("-" * 60)
    print("▶ Profiler Settings:")
    print(f"  - Use Profiler: {'✅' if config.use_profiler else '❌'}")
    if config.use_profiler:
        print(f"    - Record Shapes: {'✅' if config.record_shapes else '❌'}")
        print(f"    - Profile Memory: {'✅' if config.profile_memory else '❌'}")
        print(f"    - With Stack: {'✅' if config.with_stack else '❌'}")
        print(f"    - With FLOPs: {'✅' if config.with_flops else '❌'}")
        print(f"    - With Modules: {'✅' if config.with_modules else '❌'}")
    print("=" * 60)


def print_system_info():
    """Prints system information like PyTorch and CUDA versions."""
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA available: ✅")
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA available: ❌")
    try:
        import flash_attn

        print(f"flash_attn installed: ✅ (version: {flash_attn.__version__})")
    except ImportError:
        print("flash_attn installed: ❌")
    print("-" * 60)


# --- Model Loading ---


def get_quantization_config(config: BenchmarkConfig) -> BitsAndBytesConfig | None:
    """Builds the quantization config based on benchmark settings."""
    if not config.use_quantize:
        return None
    if config.use_int8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    else:  # 4-bit NF4
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )


def load_model_and_tokenizer(
    model_name: str, config: BenchmarkConfig, is_assistant=False
):
    """Loads a model and its tokenizer with specified configurations."""
    print(f"Loading {'assistant' if is_assistant else 'main'} model: {model_name}...")

    bnb_config = get_quantization_config(config)
    attn_implementation = (
        "flash_attention_2" if config.use_flash_attention_2 else "eager"
    )
    dtype = torch.bfloat16 if not config.use_int8 else torch.float16

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=config.device_map,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"-> Loaded in {end_time - start_time:.3f} sec")

    return model, tokenizer


# --- Data Preparation ---


def prepare_prompts(config: BenchmarkConfig, tokenizer):
    """Prepares the list of prompts from a dataset or a simple string."""
    if config.random_prompt_length > 0:

        def generate_random_string(length):
            return "".join(
                random.choices(string.ascii_letters + string.digits, k=length)
            )

        print(
            f"\n[Random Prompt Mode] Generating random prompts with length {config.random_prompt_length}"
        )
        prompts = []
        for _ in range(config.num_samples):
            prompt = ""
            current_tokens = 0
            while current_tokens < config.random_prompt_length:
                prompt += generate_random_string(
                    config.random_prompt_length - current_tokens
                )
                current_tokens = len(tokenizer.encode(prompt))
            prompts.append(prompt)
        return prompts

    if config.simple_prompt:
        print(f"\n[Simple Prompt Mode] Using prompt: '{config.simple_prompt}'")
        return [config.simple_prompt] * config.num_samples

    print(f"\n[Dataset Mode] Loading dataset: {config.dataset_name}")
    try:
        dataset = load_dataset(config.dataset_name, split="train")
        num_samples = min(config.num_samples, len(dataset))
        selected_samples = dataset.select(range(num_samples))
        print(f"-> Loaded {len(selected_samples)} samples from the dataset.")
        return [s["messages"][0]["content"] for s in selected_samples]
    except Exception as e:
        print(f"Error loading dataset {config.dataset_name}: {e}")
        raise


# --- Profiler Setup ---


def get_profiler_context(config: BenchmarkConfig):
    """Configures and returns a PyTorch profiler context manager."""
    if not config.use_profiler:
        return contextlib.nullcontext()

    log_dir = "./profile-llama3-inference"
    config_parts = [
        f"samples{config.num_samples}",
        f"batch{config.batch_size}",
        f"maxtokens{config.max_new_tokens}",
        "quant" if config.use_quantize else "no-quant",
        "flash" if config.use_flash_attention_2 else "no-flash",
        "spec" if config.use_speculative_decoding else "no-spec",
        "large-spec" if config.use_large_speculative_decoding else "no-large-spec",
        "simple" if config.simple_prompt else "dataset",
    ]
    worker_name = "_".join(config_parts)

    print(f"\n[Profiling] Logs will be saved to '{log_dir}/{worker_name}'")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
    profile_args = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "schedule": schedule,
        "on_trace_ready": torch.profiler.tensorboard_trace_handler(
            log_dir, worker_name=worker_name
        ),
        "record_shapes": config.record_shapes,
        "profile_memory": config.profile_memory,
        "with_stack": config.with_stack,
        "with_flops": config.with_flops,
        "with_modules": config.with_modules,
    }
    return torch.profiler.profile(**profile_args)


# --- Main Inference Logic ---


def run_inference(
    config: BenchmarkConfig, model, tokenizer, prompts, assistant_model=None
):
    """Runs the main inference loop and returns results and statistics."""
    print("\n[Inference Phase]")
    print("Starting inference...")

    all_results = []
    total_inference_time = 0
    total_generated_tokens = 0

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    profile_ctx = get_profiler_context(config)

    with profile_ctx as prof:
        num_batches = math.ceil(len(prompts) / config.batch_size)
        for i in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = i * config.batch_size
            batch_end = min(batch_start + config.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Format and tokenize
            messages = [[{"role": "user", "content": p}] for p in batch_prompts]
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]

            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            ).to(model.device)

            input_token_counts = [
                torch.sum(inputs.attention_mask[j]).item()
                for j in range(len(batch_prompts))
            ]

            # Inference
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs_token = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,  # Deterministic output
                    temperature=None,
                    top_p=None,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                    assistant_model=assistant_model,
                    num_assistant_tokens=config.num_assistant_tokens,
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            total_inference_time += inference_time

            # Process outputs
            for j in range(len(outputs_token)):
                total_tokens = len(outputs_token[j])
                generated_tokens = (
                    sum(
                        t != tokenizer.pad_token_id
                        for t in outputs_token[j][input_token_counts[j] :]
                    )
                    .sum()
                    .item()
                )
                total_generated_tokens += generated_tokens

                # Get the actual list of input and outpu token IDs (un-padded)
                input_token_count = input_token_counts[j]
                # input_ids_list = inputs['input_ids'][j][:input_token_count].tolist()
                output_ids_list = outputs_token[j][input_token_count:].tolist()
                assistant_response = tokenizer.decode(
                    output_ids_list, skip_special_tokens=True
                ).strip()

                all_results.append(
                    {
                        "sample_index": batch_start + j,
                        # Truncate prompt to 200 chars
                        "prompt": batch_prompts[j][:200] + "..."
                        if len(batch_prompts[j]) > 200
                        else batch_prompts[j],
                        # Truncate response to 300 chars
                        "response": assistant_response[:300] + "..."
                        if len(assistant_response) > 300
                        else assistant_response,
                        "generated_tokens": generated_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            if config.use_profiler:
                prof.step()

    print("Inference complete.")

    stats = {
        "total_inference_time_sec": total_inference_time,
        "total_generated_tokens": total_generated_tokens,
        "num_samples_processed": len(all_results),
        "tokens_per_second": total_generated_tokens / total_inference_time
        if total_inference_time > 0
        else 0,
        "throughput_samples_per_sec": len(all_results) / total_inference_time
        if total_inference_time > 0
        else 0,
    }

    return all_results, stats


# --- Reporting ---


def print_summary(
    config: BenchmarkConfig, stats: dict, memory_usage: dict, gpu_devices: str
):
    """Prints a summary of the benchmark results."""
    print(f"\n{'=' * 60}")
    print("[Overall Statistics]")
    print(f"> Main Model: {config.model_name}")
    if config.use_speculative_decoding:
        print(f"> Speculative Decoding: Enabled with {config.draft_name}")
    print(f"> GPU Devices Used: {gpu_devices}")
    print(f"> Total inference time: {stats['total_inference_time_sec']:.3f} sec")
    print(f"> Total generated tokens: {stats['total_generated_tokens']} tokens")
    print(f"> Average tokens per second: {stats['tokens_per_second']:.3f} tokens/sec")
    print(f"> Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"> Peak GPU Memory Allocated: {memory_usage['peak_mem_gb']:.2f} GB")
    print(f"{'=' * 60}")


def save_results(
    config: BenchmarkConfig,
    stats: dict,
    results: list,
    memory_usage: dict,
    gpu_devices: str,
):
    """Saves the configuration and results to a JSON file."""
    output_data = {
        "config": asdict(config),
        "gpu_devices": gpu_devices,
        "stats": stats,
        "memory_usage_gb": memory_usage,
        "results": results,
    }

    # Create a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"./results/benchmark_results_{timestamp}.json"

    try:
        os.makedirs("results", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Results saved to '{filename}'")
    except IOError as e:
        print(f"Error saving results to file: {e}")


# --- Main Execution ---


def main():
    """Main function to run the benchmark."""
    config = parse_args()
    print_config(config)
    print_system_info()

    # --- Load Models ---
    model, tokenizer = load_model_and_tokenizer(config.model_name, config)
    print(f"Model: {model.name_or_path}")
    assistant_model = None
    if config.use_speculative_decoding:
        assistant_model, _ = load_model_and_tokenizer(
            config.draft_name, config, is_assistant=True
        )
        print(f"Assistant Model: {assistant_model.name_or_path}")

    # Get a more detailed device map
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        # Get unique device IDs, filter for integers (GPU IDs), and sort them
        gpu_ids = sorted(
            list(set(v for v in device_map.values() if isinstance(v, int)))
        )
        model_devices_str = ", ".join([f"cuda:{i}" for i in gpu_ids])
        if not model_devices_str:  # Handle cases like CPU
            model_devices_str = str(model.device)
    else:
        model_devices_str = str(model.device)

    print(f"Model is running on devices: {model_devices_str}")

    # --- Warmup ---
    print("\n[Warmup Phase]")
    try:
        warmup_cnt = 10
        for _ in tqdm(range(warmup_cnt), desc="Warmup Steps"):
            warmup_prompt = "Hello, how are you?"
            warmup_inputs = tokenizer([warmup_prompt], return_tensors="pt").to(
                model.device
            )
            with torch.no_grad():
                model.generate(
                    **warmup_inputs,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                )
        torch.cuda.empty_cache()
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return  # Exit if warmup fails

    # --- Prepare Data ---
    prompts = prepare_prompts(config, tokenizer)

    # --- Run Inference ---
    if torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=i)
    else:
        torch.cuda.reset_peak_memory_stats()
    results, stats = run_inference(config, model, tokenizer, prompts, assistant_model)

    # --- Collect Memory Stats ---
    if torch.cuda.device_count() > 1:
        peak_memory_bytes = sum(
            torch.cuda.max_memory_allocated(device=i)
            for i in range(torch.cuda.device_count())
        )
    else:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
    memory_usage = {
        "peak_mem_bytes": peak_memory_bytes,
        "peak_mem_gb": peak_memory_bytes / (1024**3),
    }

    # --- Report and Save ---
    print_summary(config, stats, memory_usage, model_devices_str)
    save_results(config, stats, results, memory_usage, model_devices_str)

    if config.use_profiler:
        print(
            "\n[Profiling Complete] To view, run: tensorboard --logdir=./profile-llama3-inference"
        )


if __name__ == "__main__":
    main()
