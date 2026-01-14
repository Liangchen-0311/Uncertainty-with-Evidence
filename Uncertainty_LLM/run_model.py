import argparse
import json
import numpy as np
import re
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
import metrics 

# 参数设置
os.environ["VLLM_USE_V1"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--alias", type=str, required=True, help="name for file saving, e.g., 'llama3.1_8b_mmlu'")
args = parser.parse_args()


class SaveRawLogitsProcessor:
    def __init__(self):
        self.captured_logits = None
    def __call__(self, token_ids, logits):
        if self.captured_logits is None:
            self.captured_logits = logits.detach().float().cpu().numpy()
        return logits

def extract_answer(text):
    """
    Extract answer for MMLU (Expects A, B, C, D)
    """
    text = text.strip().upper()
    
    if not text:
        return "UNKNOWN"
    
    # MMLU typically has 4 options: A, B, C, D
    if text in ["A", "B", "C", "D"]:
        return text
    
    import re
    match = re.search(r'^([A-D])', text)
    if match:
        return match.group(1)
        
    number_map = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D'
    }
    if text in number_map:
        return number_map[text]
        
    return "UNKNOWN"


def load_mmlu_data(num_samples, subset_name="college_mathematics", split="test"):
    """
    Load MMLU Dataset
    subset_name examples: 'abstract_algebra', 'college_mathematics', 'professional_law', 'high_school_physics'
    """
    print(f"Loading MMLU (Subset: {subset_name}, Split: {split})...")
    
    try:
        # Load from Hugging Face (cais/mmlu)
        ds = load_dataset("cais/mmlu", subset_name, split=split)
    except Exception as e:
        print(f"Error loading MMLU: {e}. Check internet connection or subset name.")
        return []
    
    formatted_data = []
    
    # MMLU answers are 0, 1, 2, 3 -> Map to A, B, C, D
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    target_count = min(num_samples, len(ds))
    
    for i in range(target_count):
        item = ds[i]
        
        choices_text = item['choices']
        # MMLU always has 4 options
        choices_label = ['A', 'B', 'C', 'D']
        
        answer_idx = item['answer']
        ground_truth = idx_to_letter.get(answer_idx, "UNKNOWN")
            
        formatted_data.append({
            "question": item['question'],
            "choices_text": choices_text,   
            "choices_label": choices_label, 
            "answer": ground_truth,
            "subject": subset_name
        })
        
    print(f"Successfully loaded {len(formatted_data)} samples from MMLU ({subset_name}).")
    return formatted_data

def get_few_shot_prompt(target_question, target_choices_text, target_choices_label):
    """
    Generate Few-Shot Prompt for MMLU (Academic Style)
    """
    # Few-shot examples 
    examples = [
        {
            "q": "What is the capital of France?",
            "opts": ["London", "Berlin", "Paris", "Madrid"],
            "labels": ["A", "B", "C", "D"],
            "a": "C"
        },
        {
            "q": "Which element has the atomic number 1?",
            "opts": ["Helium", "Hydrogen", "Lithium", "Carbon"],
            "labels": ["A", "B", "C", "D"],
            "a": "B"
        },
        {
            "q": "If x + 2 = 5, what is the value of x?",
            "opts": ["1", "2", "3", "4"],
            "labels": ["A", "B", "C", "D"],
            "a": "C"
        }
    ]
    
    prompt_text = "The following are multiple choice questions (with answers) about academic subjects.\n\n"

    # Append Examples
    for ex in examples:
        prompt_text += f"Question: {ex['q']}\nOptions:\n"
        for label, text in zip(ex['labels'], ex['opts']):
            prompt_text += f"{label}. {text}\n"
        prompt_text += f"Answer: {ex['a']}\n\n"
    
    prompt_text += f"Question: {target_question}\nOptions:\n"
    for label, text in zip(target_choices_label, target_choices_text):
        prompt_text += f"{label}. {text}\n"
        
    prompt_text += "Answer: "
    
    return prompt_text


# 主程序
def main():
    print(f"Loading Model: {args.model_name} ...")

    llm = LLM(model=args.model_name, trust_remote_code=True, tensor_parallel_size=1, dtype="bfloat16") 
    tokenizer = llm.get_tokenizer()

    # Define allowed tokens for MMLU
    target_tokens = ["A", "B", "C", "D", " A", " B", " C", " D"]
    target_tokens += ["1", "2", "3", "4", " 1", " 2", " 3", " 4"]
    
    allowed_ids = []
    for t in target_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            allowed_ids.append(ids[-1])

    allowed_ids = list(set(allowed_ids))
    print(f"Constrained decoding to IDs: {allowed_ids}")

    # 计算指标
    K_VALUE = 4  # 可以根据需要调整 K 值
    calc_eu = metrics.get_eu("eu", k=K_VALUE)
    calc_au = metrics.get_eu("au", k=K_VALUE)
    calc_entropy = metrics.get_eu("entropy")
    calc_prob = metrics.get_eu("prob")

    # Load MMLU Data

    dataset = load_mmlu_data(num_samples=200, subset_name="high_school_mathematics", split="test")

    results = []
    print("Starting Inference...")

    for i, item in enumerate(dataset):
        
        prompt = get_few_shot_prompt(item['question'], item['choices_text'], item['choices_label'])
        
        logits_saver = SaveRawLogitsProcessor()
        sampling_params = SamplingParams(
            temperature=1, 
            max_tokens=1,
            logprobs=None, 
            logits_processors=[logits_saver],
            allowed_token_ids=allowed_ids,
        )

        try:
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

            captured = logits_saver.captured_logits

            if captured.ndim == 2:
                raw_logits = captured[0]
            elif captured.ndim == 1:
                raw_logits = captured
            else:
                print(f"Unexpected shape: {captured.shape}")
                continue

            pred_text = outputs[0].outputs[0].text

            # Calculate Metrics
            eu = float(calc_eu(raw_logits))
            au = float(calc_au(raw_logits))
            entropy = float(calc_entropy(raw_logits))
            prob = float(calc_prob(raw_logits))

            # Reliability Product
            product_score = - (eu + au)

            is_correct = (extract_answer(pred_text) == item['answer'])
            
            # Print progress every 50 samples
            if i % 50 == 0:
                print(f"Sample {i}: Correct={is_correct}, EU={eu:.4f}, AU={au:.4f}")

            results.append({
                "is_correct": is_correct,
                "score_prob": prob,          
                "score_entropy": -entropy,   
                "score_product": product_score,
                "score_eu": -eu,
                "score_au": -au
            })

        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    # Save Results
    filename = f"result_{args.alias}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Done! Saved to {filename}")

if __name__ == "__main__":
    main()