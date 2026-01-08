from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokens():
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name: str = "llm-jp/Synthetic-JP-EN-Coding-Dataset"
    num_samples: int = 512

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset(dataset_name, split="train")
    selected_samples = dataset.select(range(num_samples))
    print(f"-> Loaded {len(selected_samples)} samples from the dataset.")

    input_prompts = [s["messages"][0]["content"] for s in selected_samples]
    input_tokens_list = []
    for s in input_prompts:
        formatted_s = tokenizer.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(formatted_s)
        input_tokens_list.append(len(tokens))

    output_prompts = [s["messages"][1]["content"] for s in selected_samples]
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
