# main_short.py
import os
import re
import json
import math
import fire
import vllm
from tqdm.auto import tqdm
import unicodedata
from typing import Optional
from src.utils import build_short_answer_inputs
from src.datasets.gpqa import GPQADataset
from src.datasets.arc import ARCDataset
from src.datasets.bigbench import BigBenchDataset
from src.datasets.commonsense import CommonsenseQADataset
from src.datasets.musr import MuSRMurderMysteriesBinaryDataset

def normalize_text_sa(s: Optional[str]) -> str:

    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r'^[\s\'"(\[{<]+', '', s)
    s = re.sub(r'[\s\'"\])}>.,;:!?]+$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.casefold()
    return s

def extract_sa_from_output(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    lower = text.lower()
    think_idx = lower.rfind('</think>')
    scope = text[think_idx + len('</think>'):] if think_idx != -1 else text

    matches = re.findall(r'<\s*([^<>]+?)\s*>', scope)
    if matches:
        cand = matches[-1].strip()
        if cand and cand.lower() != 'think':
            return cand

    return None

def main(
    reasoning: bool = False,
    output_file: str = "./results/short_answer.json",
    base_model_id: str = "microsoft/Phi-4-reasoning",
    dtype: str = "float16",
    quantization: str = "awq",
    max_model_len: int = 8192,
    batch_size: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    dataset_name: str = "gpqa",  
    split: str = "train",
    limit: int = 3,
    prompt_cot: str = "da",    
):
    print(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES','(not set)')}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting...")
        return

    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=42,
    )

    llm = vllm.LLM(
        model=base_model_id,
        task="generate",
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )

    ds_key = dataset_name.lower()
    if ds_key in {"gpqa", "gpqa_main"}:
        adapter = GPQADataset(split=split, config="gpqa_main")
    elif ds_key in {"arc", "arc_challenge"}:
        adapter = ARCDataset(split="validation", subset="ARC-Challenge")
    elif ds_key in {"bigbench", "bigbench_anu"}:
        adapter = BigBenchDataset()
    elif ds_key in {"commonsense", "commonsense_qa"}:
        adapter = CommonsenseQADataset()
    elif ds_key in {"musr", "musr_murder_mysteries_binary"}:
        adapter = MuSRMurderMysteriesBinaryDataset()
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # build_inputs는 (input_list, answer_labels, ds, rows) 반환 
    _, _, dataset, rows = adapter.build_inputs(
        limit=limit, prompt_cot="da", seed=42, show_tqdm=True, include_rows=True
    )

    # Short Answer 프롬프트 생성 
    input_list, gold_answers, metas = build_short_answer_inputs(
        rows, prompt_cot=prompt_cot
    )
    outputs = metas  

    # 생성 & 평가
    total_batches = max(1, math.ceil(len(input_list) / batch_size))
    for start in tqdm(range(0, len(input_list), batch_size),
                      desc=f"Generating Short Answers ({dataset_name})",
                      total=total_batches):
        end = min(start + batch_size, len(input_list))
        batch_prompts = input_list[start:end]

        result = llm.chat(batch_prompts, sampling_params=sampling_params, use_tqdm=False)

        for k in range(end - start):
            idx = start + k
            gen_text = result[k].outputs[0].text

            pred_raw = extract_sa_from_output(gen_text)

            # 정규화
            pred_norm = normalize_text_sa(pred_raw)
            gold_norm = normalize_text_sa(gold_answers[idx])

            is_correct = (pred_norm != "" and pred_norm == gold_norm)

            outputs[idx].update({
                "generated_text": gen_text,
                "prediction_raw": pred_raw,
                "prediction_norm": pred_norm,
                "gold_norm": gold_norm,
                "is_correct": is_correct,
                "output_tkn": len(result[k].outputs[0].token_ids),
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    # 결과 통계
    total = len(outputs)
    correct = sum(1 for o in outputs if o.get("is_correct"))
    acc = correct / total if total > 0 else 0.0
    tkn_sum = sum(o.get("output_tkn", 0) for o in outputs if o.get("output_tkn") is not None)
    tkn_cnt = sum(1 for o in outputs if o.get("output_tkn") is not None)
    avg_tkn = (tkn_sum / tkn_cnt) if tkn_cnt > 0 else 0.0

    print(f"\n=== ShortAnswer({dataset_name}) with {base_model_id} ===")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy:{acc:.4f}")
    print(f"Avg output tokens (TKN): {avg_tkn:.2f}")
    print(f"Results(json): {output_file}")


if __name__ == "__main__":
    fire.Fire(main)