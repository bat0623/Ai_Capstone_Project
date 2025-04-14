import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    # Base model and LoRA adapter paths
    base_model_name = "beomi/polyglot-ko-3.8b"
    adapter_path = "./finetuned-polyglot"  # or e.g. "./finetuned-polyglot/checkpoint-XXXX" for a specific checkpoint
    # Load the base model (with half precision on GPU)
    print(f"Loading base model '{base_model_name}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Load the fine-tuned LoRA adapter into the base model
    print(f"Loading LoRA adapter from '{adapter_path}'...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    print("Model loaded. Enter a prompt to generate a response (Ctrl+C to exit).")
    # Interactive loop for user prompts
    while True:
        try:
            user_input = input("Prompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting inference.")
            break
        if not user_input:
            continue  # skip empty input
        # Tokenize user prompt
        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        # Generate a response from the model
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # The output sequence includes the prompt followed by the generated response
        generated_sequence = output_ids[0]
        # Slice the generated tokens to get only the new response (excluding the prompt tokens)
        generated_tokens = generated_sequence[inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Response: {generated_text.strip()}")

if __name__ == "__main__":
    main()
