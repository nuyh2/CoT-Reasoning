import os
import re
import json
import random
import dotenv
dotenv.load_dotenv()

import math
import fire
import vllm
import pandas as pd
import copy
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
    
def extract_label_from_output(str):
    # </think> 뒤에 혹은 <> 사이에 오는 ABCD 추출
    match = re.search(r'</think>\s*([A-D])', str)
    if match:
        return match.group(1)
    
    match = re.search(r'<\s*([A-D])\s*>', str)
    if match:
        return match.group(1)
    
    return None

# 프롬프트 template 생성 함수
def build_template(question: str, option_a: str, option_b: str, option_c: str, option_d: str):
    system = (
        "You are a helpful assistant."
    )

    user = (
        "Answer the following question with the given context.\n\n"
        f"Question: {question}\n\n"
        "Answer Choices:\n"
        f"A: {option_a}\n"
        f"B: {option_b}\n"
        f"C: {option_c}\n"
        f"D: {option_d}\n\n"
        "Output only the correct label as a single character. Do not add any extra text.\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# 프롬프트 template 생성 함수
def build_reprediction_template(question: str, option_a: str, option_b: str, option_c: str, option_d: str, first_prediction_cot: str = ""):
    prompt_text = (
        "Answer the following question with the given context.\n\n"
        f"Question: {question}\n\n"
        "Answer Choices:\n"
        f"A: {option_a}\n"
        f"B: {option_b}\n"
        f"C: {option_c}\n"
        f"D: {option_d}\n\n"
        "Let's think step by step.\n"
        "After reasoning, output the correct label on the last line, between '<' and '>'. Output format example: <label>\n\n"
        f"{first_prediction_cot}\nTherefore, the answer is <"
        # 모델이 <~>를 html 태그로 헷갈리기가 쉬워서, 좀 더 명확히 instruction을 줬습니다 <와 > 사이에 생성하라고
    )

    return prompt_text

def main(
    output_file: str = "./results/GPQA_phi4_reasoning.json",
    base_model_id: str = "microsoft/Phi-4-reasoning",
    dtype: str = "float16",
    batch_size: int = 16,
    max_tokens: int = 2048,
    max_model_len: int = 8192,
    reasoning: bool = False,
    quantization: str = "awq",
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    print(f"CUDA visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting...")
        return

    # GPQA 데이터셋 로딩 및 필요한 Column 추출
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
    
    dataset = dataset["train"].select_columns(["Question", "Correct Answer", 
                                      "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", 
                                      "Subdomain", "High-level domain"])

    # 생성 파라미터 설정
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=42,
    )

    # 모델 선언
    llm = vllm.LLM(
        model=base_model_id,
        task="generate",
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        enforce_eager = True,
    )

    # 입력 데이터 형태 구성
    input_list = []
    answer_labels = []
    outputs = []
    for i in tqdm(range(len(dataset)), desc=f"Building Prompts", total=len(dataset)):
        question = dataset[i]['Question']
        
        option = []
        option.append(dataset[i]['Correct Answer'])
        option.append(dataset[i]['Incorrect Answer 1'])
        option.append(dataset[i]['Incorrect Answer 2'])
        option.append(dataset[i]['Incorrect Answer 3'])
        random.shuffle(option)

        input_list.append(build_template(question, option[0], option[1], option[2], option[3]))
        
        if option[0]==dataset[i]['Correct Answer']:
            answer_labels.append('A')
        elif option[1]==dataset[i]['Correct Answer']:
            answer_labels.append('B')
        elif option[2]==dataset[i]['Correct Answer']:
            answer_labels.append('C')
        elif option[3]==dataset[i]['Correct Answer']:
            answer_labels.append('D')
        
        outputs.append({
            'question': question,
            'label_A': option[0],
            'label_B': option[1],
            'label_C': option[2],
            'label_D': option[3],
            'answer': answer_labels[-1],
        })
    
    # 모델 생성 및 결과물 저장
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating Responses", total=len(dataset)//batch_size):
        batch_prompt = input_list[i:i+batch_size]
        
        output = llm.chat(batch_prompt, sampling_params=sampling_params, use_tqdm=False)
        
        for j in range(i, min(i+batch_size, len(dataset))):
            generated = output[j-i].outputs[0].text
            pred = extract_label_from_output(generated)
            
            outputs[j].update({
                'generated_text': generated,
                'prediction': pred,
                "is_correct": (pred==answer_labels[j]),
                "output_tkn": len(output[j-i].outputs[0].token_ids),
                "repredicted": False,
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
    
    # Second Prediction 생성
    reprediction_target_idx = []
    reprediction_input_list = []
    
    reprediction_params = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=12, seed=42
    )
    
    for i, o in enumerate(outputs):
        if o.get("prediction") is None:
            reprediction_target_idx.append(i)
    
    print(f"\nreprediction target: {reprediction_target_idx}\n")
    
    for i in tqdm(reprediction_target_idx,
                  desc=f"Building Reprediction Prompts",
                  total=len(reprediction_target_idx)):
        reprediction_input_list.append(build_reprediction_template(outputs[i]['question'],
                                                                   outputs[i]['label_A'],
                                                                   outputs[i]['label_B'],
                                                                   outputs[i]['label_C'],
                                                                   outputs[i]['label_D'],
                                                                   outputs[i]['generated_text']))

    for start in tqdm(range(0, len(reprediction_input_list), batch_size),
                  desc=f"Generating Reprediction Responses",
                  total=math.ceil(len(reprediction_input_list)/batch_size)):
        end = min(start + batch_size, len(reprediction_input_list))
        batch_prompt = reprediction_input_list[start:end]
        
        outputs_batch = llm.generate(batch_prompt, sampling_params=reprediction_params, use_tqdm=False)
        batch_indices = reprediction_target_idx[start:end]
        
        for k, idx in enumerate(batch_indices):
            generated_prev = outputs[idx]['generated_text']
            generated_cur = outputs_batch[k].outputs[0].text
            print(f"{generated_cur}\n")
            generated = generated_prev + "Therefore, the answer is <" + generated_cur
            
            pred = extract_label_from_output(generated)
            
            output_tkn_prev = outputs[idx].get("output_tkn", 0)
            output_tkn_cur = len(outputs_batch[k].outputs[0].token_ids)
            
            outputs[idx].update({
                'generated_text': generated,
                'prediction': pred,
                "is_correct": (pred == answer_labels[idx]),
                "output_tkn": output_tkn_prev + output_tkn_cur,
                "repredicted": True,
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

    total = len(outputs)
    correct = sum(1 for o in outputs if o["is_correct"])
    accuracy = correct / total if total > 0 else 0.0

    tkn_sum = sum(o["output_tkn"] for o in outputs if o["output_tkn"] is not None)
    tkn_count = sum(1 for o in outputs if o["output_tkn"] is not None)
    avg_tkn = (tkn_sum / tkn_count) if tkn_count > 0 else 0.0

    print(f"\n=== GPQA(train) with {base_model_id} ===")
    print(f"Reasoning mode: {reasoning}")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy:{accuracy:.4f}")
    print(f"Avg output tokens (TKN): {avg_tkn:.2f}")
    print(f"Results(jsonl): {output_file}")

    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "res.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== GPQA(train) with {base_model_id} ===")
        f.write(f"Reasoning mode: {reasoning}")
        f.write(f"Total:   {total}")
        f.write(f"Correct: {correct}")
        f.write(f"Accuracy:{accuracy:.4f}")
        f.write(f"Avg output tokens (TKN): {avg_tkn:.2f}")
        f.write(f"Results(jsonl): {output_file}")
        f.write("-" * 50 + "\n")

if __name__ == "__main__":
    fire.Fire(main)
