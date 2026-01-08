import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- Configuration & Argument Parsing ---


@dataclass
class VLLMBenchmarkConfig:
    """Configuration class for the vLLM inference benchmark."""

    # Model and Dataset
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name: str = "llm-jp/Synthetic-JP-EN-Coding-Dataset"
    quantization: str = None  # vLLM-specific quantization (e.g., "awq", "gptq", "fp8")

    # Inference Parameters
    num_samples: int = 128
    max_new_tokens: int = 1000
    simple_prompt: str = None

    # vLLM specific parameters
    tensor_parallel_size: int = 1  # Number of GPUs to use

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tensor_parallel_size > torch.cuda.device_count():
            raise ValueError(
                f"Tensor parallel size ({self.tensor_parallel_size}) cannot be "
                f"greater than the number of available GPUs ({torch.cuda.device_count()})."
            )


def parse_args() -> VLLMBenchmarkConfig:
    """Parses command-line arguments and returns a VLLMBenchmarkConfig instance."""
    parser = argparse.ArgumentParser(
        description="LLM inference benchmark script using vLLM."
    )

    # Dynamically add arguments from the BenchmarkConfig dataclass
    for field in VLLMBenchmarkConfig.__dataclass_fields__.values():
        arg_name = f"--{field.name.replace('_', '-')}"
        parser.add_argument(
            arg_name,
            type=field.type,
            default=field.default,
            help=f"{field.metadata.get('help', '')} (default: {field.default})",
        )

    args = parser.parse_args()
    return VLLMBenchmarkConfig(**vars(args))


def print_config(config: VLLMBenchmarkConfig):
    """Prints the benchmark configuration in a readable format."""
    print("=" * 60)
    print("🚀 Benchmarking Llama 3 Inference with vLLM")
    print("-" * 60)
    print("▶ General Settings:")
    print(f"  - Model: {config.model_name}")
    print(f"  - Number of samples: {config.num_samples}")
    print(f"  - Max new tokens: {config.max_new_tokens}")
    if config.simple_prompt:
        print("▶ Prompt Source: Simple Prompt Mode ✅")
        print(f"  - Prompt: '{config.simple_prompt}'")
    else:
        print(f"▶ Prompt Source: Dataset Mode ({config.dataset_name})")
    print("-" * 60)
    print("▶ vLLM Settings:")
    print(f"  - Quantization: {config.quantization or 'None'}")
    print(f"  - Tensor Parallel Size (GPUs): {config.tensor_parallel_size}")
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
        import vllm

        print(f"vllm installed: ✅ (version: {vllm.__version__})")
    except ImportError:
        print("vllm installed: ❌")
    print("-" * 60)


# --- Model Loading ---


def load_vllm_model_and_tokenizer(config: VLLMBenchmarkConfig):
    """Loads a vLLM engine and a tokenizer for prompt formatting."""
    print(f"Loading model with vLLM: {config.model_name}...")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    try:
        llm = LLM(
            model=config.model_name,
            quantization=config.quantization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as e:
        print(f"Error loading model {config.model_name} with vLLM: {e}")
        raise

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"-> Loaded in {end_time - start_time:.3f} sec")

    return llm, tokenizer


# --- Data Preparation ---


def prepare_prompts(config: VLLMBenchmarkConfig):
    """Prepares the list of prompts from a dataset or a simple string."""
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


# --- Main Inference Logic ---


def run_vllm_inference(config: VLLMBenchmarkConfig, llm: LLM, tokenizer, prompts: list):
    """Runs the main inference loop using vLLM and returns results and statistics."""
    print("\n[Inference Phase]")
    print("Formatting prompts with chat template...")

    # Format and tokenize
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    # setting up the sampling parameters for vLLM
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,  # Deterministic output
        top_p=1.0,
        max_tokens=config.max_new_tokens,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    print("Starting inference...")
    # Inference with vLLM (once per prompt list)
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    request_outputs = llm.generate(formatted_prompts, sampling_params)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_inference_time = end_time - start_time
    total_generated_tokens = sum(len(o.outputs[0].token_ids) for o in request_outputs)

    print("Inference complete.")

    # Process outputs
    all_results = []
    for i, output in enumerate(request_outputs):
        prompt_text = prompts[i]
        generated_text = output.outputs[0].text

        all_results.append(
            {
                "sample_index": i,
                "prompt": prompt_text[:200] + "..."
                if len(prompt_text) > 200
                else prompt_text,
                "response": generated_text[:300] + "..."
                if len(generated_text) > 300
                else generated_text,
                "generated_tokens": len(output.outputs[0].token_ids),
                "prompt_tokens": len(output.prompt_token_ids),
            }
        )

    stats = {
        "total_inference_time_sec": total_inference_time,
        "total_generated_tokens": total_generated_tokens,
        "num_samples_processed": len(request_outputs),
        "tokens_per_second": total_generated_tokens / total_inference_time
        if total_inference_time > 0
        else 0,
        "throughput_samples_per_sec": len(request_outputs) / total_inference_time
        if total_inference_time > 0
        else 0,
    }

    return all_results, stats


# --- Reporting ---


def print_summary(config: VLLMBenchmarkConfig, stats: dict, memory_usage: dict):
    """Prints a summary of the benchmark results."""
    print(f"\n{'=' * 60}")
    print("[Overall Statistics]")
    print(f"> Model: {config.model_name}")
    print(f"> Quantization: {config.quantization or 'None'}")
    print(f"> GPU Devices Used: {config.tensor_parallel_size}")
    print(f"> Total inference time: {stats['total_inference_time_sec']:.3f} sec")
    print(f"> Total generated tokens: {stats['total_generated_tokens']} tokens")
    print(f"> Average tokens per second: {stats['tokens_per_second']:.3f} tokens/sec")
    print(f"> Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")
    if memory_usage["peak_mem_gb"] is None:
        print("> Peak GPU Memory (PyTorch): N/A GB")
    else:
        print(f"> Peak GPU Memory (PyTorch): {memory_usage['peak_mem_gb']:.2f} GB")
    print(f"{'=' * 60}")


def save_results(
    config: VLLMBenchmarkConfig, stats: dict, results: list, memory_usage: dict
):
    """Saves the configuration and results to a JSON file."""
    output_data = {
        "config": asdict(config),
        "stats": stats,
        "memory_usage_gb": memory_usage,
        "results": results,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"./results/vllm_benchmark_results_{timestamp}.json"

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

    # --- Load Model ---
    llm, tokenizer = load_vllm_model_and_tokenizer(config)

    # --- Warmup ---
    print("\n[Warmup Phase]")
    try:
        warmup_prompt = "Hello, how are you?"
        warmup_formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": warmup_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        warmup_params = SamplingParams(max_tokens=10)
        llm.generate(warmup_formatted_prompt, warmup_params)
        torch.cuda.empty_cache()
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # --- Prepare Data ---
    prompts = prepare_prompts(config)

    # --- Run Inference ---
    # torch.cuda.reset_peak_memory_stats()  # comment out because vLLM doesn't collect this
    results, stats = run_vllm_inference(config, llm, tokenizer, prompts)

    # --- Collect Memory Stats ---
    # peak_memory_bytes = torch.cuda.max_memory_allocated()  # comment out because vLLM doesn't collect this
    memory_usage = {
        "peak_mem_bytes": None,
        "peak_mem_gb": None,
    }

    # --- Report and Save ---
    print_summary(config, stats, memory_usage)
    save_results(config, stats, results, memory_usage)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
