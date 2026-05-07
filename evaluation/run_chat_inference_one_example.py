from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = [
    "AbeerMostafa/Novelty_Reviewer",
    "Qwen/Qwen2.5-14B-Instruct-1M",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "SenthilKumarN/SciLlama-3.2-3B",
    "maxidl/Llama-OpenReviewer-8B",
    "weathon/paper_reviewer",
    "WestlakeNLP/DeepReviewer-7B"
]

def run_inference(model_name, prompt, max_new_tokens=512):
    """Run inference on a single model with the given prompt."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        
        # Try to use chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        else:
            # Fallback to simple tokenization
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            inputs = inputs["input_ids"]
        
        # Generate response
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response if it's included
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        print(f"Response:\n{response}\n")
        
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"Error with {model_name}: {str(e)}\n")
        return None


# Main execution
if __name__ == "__main__":
    # Insert your prompt here
    prompt = """review the following paper abstract for novelty: 
    
    
    
    """
    if not prompt:
        print("Please insert a prompt string before running!")
    else:
        results = {}
        for model_name in MODELS:
            response = run_inference(model_name, prompt)
            results[model_name] = response
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for model_name, response in results.items():
            print(f"{model_name}: {response}")