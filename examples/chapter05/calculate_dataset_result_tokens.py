from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokens():
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name: str = "tatsu-lab/alpaca"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(dataset_name, split="train")
    print(f"-> Loaded {len(dataset)} samples from the dataset.")

    def format_prompt(s):
        prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
                  f"Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:\n{s['instruction']}\n\n")
        if s.get("input"):
            prompt += f"### Input:\n{s['input']}\n\n"
        prompt += "### Response:\n"
        return prompt

    input_prompts = [format_prompt(s) for s in dataset]
    input_tokens_list = []
    for s in input_prompts:
        tokens = tokenizer.encode(s)
        input_tokens_list.append(len(tokens))

    output_prompts = [s["output"] for s in dataset]
    output_tokens_list = []
    for s in output_prompts:
        tokens = tokenizer.encode(s)
        output_tokens_list.append(len(tokens))

    return input_tokens_list, output_tokens_list

def main():
    input_tokens_list, output_tokens_list = get_tokens()

    print(f"Input Mean: {sum(input_tokens_list)/len(input_tokens_list)}")
    print(f"Output Mean: {sum(output_tokens_list)/len(output_tokens_list)}")

if __name__ == "__main__":
    main()
