import os
import re
import json
import math
import fire
import vllm
from tqdm.auto import tqdm
from src.utils import build_reprediction_template, build_tnf_inputs
from src.datasets.gpqa import GPQADataset
from src.datasets.arc import ARCDataset
from src.datasets.bigbench import BigBenchDataset
from src.datasets.commonsense import CommonsenseQADataset
from src.datasets.musr import MuSRMurderMysteriesBinaryDataset

def extract_label_from_output(str):
    # </think> 뒤에 혹은 <> 사이에 오는 ABCD 추출
    match = re.search(r'</think>\s*([A-D])', str)
    if match:
        return match.group(1)
    
    match = re.search(r'<\s*([A-D])\s*>', str)
    if match:
        return match.group(1)
    
    return None

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
    prompt_cot: str = "da",
    dataset_name: str = "gpqa",
):

    print(f"CUDA visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting...")
        return


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


    # ARC
    if dataset_name == 'arc':
        adapter = ARCDataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=10, prompt_cot=prompt_cot, seed=42, show_tqdm=True, include_rows=True
        )

    #Bigbench
    elif dataset_name == "bigbench":
        adapter = BigBenchDataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=10, prompt_cot=prompt_cot, seed=42, show_tqdm=True, include_rows=True
        )

    # commonsense
    elif dataset_name == "commonsense":
        adapter = CommonsenseQADataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=10, prompt_cot=prompt_cot, seed=42, show_tqdm=True, include_rows=True
        )

    # # musr
    elif dataset_name == "musr":
        adapter = MuSRMurderMysteriesBinaryDataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=10, prompt_cot=prompt_cot, seed=42, show_tqdm=True, include_rows=True
        )

    else :
        # GPQA - default
        adapter = GPQADataset(split="train")  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=10, prompt_cot=prompt_cot, seed=42, show_tqdm=True, include_rows=True
        )

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

    log_dir = "results_mcq"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "res.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== ShortAnswer({dataset_name}) with {base_model_id} ===\n")
        f.write(f"Reasoning mode: {reasoning}")
        f.write(f"Total:   {total}")
        f.write(f"Correct: {correct}")
        f.write(f"Accuracy:{accuracy:.4f}")
        f.write(f"Avg output tokens (TKN): {avg_tkn:.2f}")
        f.write(f"Results(jsonl): {output_file}")
        f.write("-" * 50 + "\n")

if __name__ == "__main__":
    fire.Fire(main)

