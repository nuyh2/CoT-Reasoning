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

def extract_tf_from_output(text: str):
    match = re.search(r'<\s*(True|False)\s*>', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    match = re.search(r'\b(True|False)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

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
            limit=3, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
        )

    #Bigbench
    elif dataset_name == "bigbench":
        adapter = BigBenchDataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=3, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
        )

    # commonsense
    elif dataset_name == "commonsense":
        adapter = CommonsenseQADataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=3, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
        )

    # # musr
    elif dataset_name == "musr":
        adapter = MuSRMurderMysteriesBinaryDataset()  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=3, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
        )

    else :
        # GPQA - default
        adapter = GPQADataset(split="train")  
        input_list, answer_labels, dataset, outputs = adapter.build_inputs(
            limit=3, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
        )


    # True/False 변환
    input_list, answer_labels, outputs = build_tnf_inputs(
        outputs, prompt_cot=prompt_cot, seed=42, p_true=0.5
    )


    # 모델 생성 및 평가
    for i in tqdm(range(0, len(input_list), batch_size),
                  desc=f"Generating Responses",
                  total=len(input_list)//batch_size):
        batch_prompt = input_list[i:i+batch_size]

        result = llm.chat(batch_prompt, sampling_params=sampling_params, use_tqdm=False)

        for j in range(i, min(i+batch_size, len(input_list))):
            generated = result[j-i].outputs[0].text
            pred = extract_tf_from_output(generated)
            gold = answer_labels[j]

            outputs[j].update({
                "generated_text": generated,
                "prediction": pred,
                "is_correct": (pred == gold),
                "output_tkn": len(result[j-i].outputs[0].token_ids),
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    
    # 최종 통계
    total = len(outputs)
    correct = sum(1 for o in outputs if o["is_correct"])
    accuracy = correct / total if total > 0 else 0.0

    tkn_sum = sum(o["output_tkn"] for o in outputs if o.get("output_tkn") is not None)
    tkn_count = sum(1 for o in outputs if o.get("output_tkn") is not None)
    avg_tkn = (tkn_sum / tkn_count) if tkn_count > 0 else 0.0

    print(f"\n=== TNF({dataset_name}) with {base_model_id} ===")
    print(f"Reasoning mode: {reasoning}")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy:{accuracy:.4f}")
    print(f"Avg output tokens (TKN): {avg_tkn:.2f}")
    print(f"Results(jsonl): {output_file}")


if __name__ == "__main__":
    fire.Fire(main)