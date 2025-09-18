# src/utils.py
from __future__ import annotations
import string
from typing import List, Dict, Any, Tuple, Optional
import random

def _make_labels(n: int) -> List[str]:
    if not (2 <= n <= 10):
        raise ValueError("Number of choices must be between 2 and 10.")
    return list(string.ascii_uppercase[:n])  

def _format_choices(choices: List[str]) -> str:
    labels = _make_labels(len(choices))
    return "\n".join(f"{labels[i]}: {choices[i]}" for i in range(len(choices)))

# 다지선다 프롬프트 생성 
def build_template_mcq(
    question: str,
    choices: List[str],
    prompt_cot: Optional[str] = None,
    allow_think_tags: bool = True,
    system_msg: str = "You are a helpful assistant.",
) -> List[dict]:
    labels = _make_labels(len(choices))
    choice_lines = _format_choices(choices)

    prompts = []
    prompts.append({"role": "system", "content": system_msg})

    user = (
        "Answer the following question with the given context.\n\n"
        f"Question: {question}\n\n"
        "Answer Choices:\n"
        f"{choice_lines}\n\n"   
    )

    # 모드 결정
    if prompt_cot == "cot":
        allowed = " / ".join(f"<{L}>" for L in labels)
        user += (
            "Let's think step by step.\n"
            "After reasoning, output the correct label on the last line, between '<' and '>'. Output format example: <label>\n\n"
        )
        prompts.append( {"role": "user", "content": user})

    else:
        user += ("Output only the correct label as a single character. Do not add any extra text.\n")
        prompts.append( {"role": "user", "content": user})

    return prompts


# reprediction - mcp
def build_reprediction_template(
    question: str,
    choices: List[str],
    first_prediction_cot: str = "",
) -> str:
    labels = _make_labels(len(choices))
    choice_lines = _format_choices(choices)
    allowed = " / ".join(f"<{L}>" for L in labels)

    prompt_text = (
        "Answer the following question with the given context.\n\n"
        f"Question: {question}\n\n"
        "Answer Choices:\n"
        f"{choice_lines}\n\n"
        "Let's think step by step.\n"
        f"After reasoning, output the correct label on the last line, between '<' and '>' in one of: {allowed}\n\n"
        f"{first_prediction_cot}\nTherefore, the answer is <"
    )
    return prompt_text


## 다지선다를 tnf 형식으로 변환
def build_tnf_inputs(
    rows: List[Dict[str, Any]],
    prompt_cot: str = "da",            
    seed: int = 42,
    p_true: float = 0.5,               
    allow_think_tags: bool = True,
    system_msg: str = "You are a helpful assistant.",
) -> Tuple[List[List[dict]], List[str], List[Dict[str, Any]]]:

    rng = random.Random(seed)

    def _available_labels(row: Dict[str, Any]) -> List[str]:
        labs = []
        for L in string.ascii_uppercase[:10]:  # A..J
            val = row.get(f"label_{L}")
            if isinstance(val, str) and val.strip():
                labs.append(L)
        return labs

    tf_inputs: List[List[dict]] = []
    tf_labels: List[str] = []
    tf_rows:   List[Dict[str, Any]] = []

    for row in rows:
        question = row.get("question", "")
        gold_letter = (row.get("answer") or "").strip().upper()

        labs = _available_labels(row)
        if len(labs) < 2 or gold_letter not in labs:
            continue

        opt_map = {L: row[f"label_{L}"] for L in labs}
        gold_text = opt_map[gold_letter]

        # 50:50로 True/False 결정
        make_true = (rng.random() < p_true)

        if make_true:
            answer_text = gold_text
            tf_gold = "True" 
        else:
            distractors = [opt_map[L] for L in labs if L != gold_letter]
            if distractors:
                answer_text = rng.choice(distractors)
                tf_gold = "False" 
            else:
                answer_text = gold_text
                tf_gold = "True"

        # 시스템 메시지
        msgs = [{"role": "system", "content": system_msg}]

        # user 메시지
        user = (
            f"Question: {question}\n"
            f"Answer: {answer_text}\n"
            "Is this answer true or false for this question? "
            "You must choose either True or False."
        )

        if prompt_cot == "cot":
            user += (
                " Let's think step by step.\n"
                "After reasoning, output 'True' or 'False' on the last line, between '<' and '>'. "
                "Output format example: <Answer>\n\n"
            )
        else:
            user += (
                " Output only the correct answer as either 'True' or 'False'. Do not add any extra text.\n"
            )

        msgs.append({"role": "user", "content": user})

        tf_inputs.append(msgs)
        tf_labels.append(tf_gold)
        tf_rows.append({
            "question": question,
            "answer_text": answer_text,
            "gold_tf_label": tf_gold,
            "is_true": (tf_gold == "True"),
            "orig_gold_letter": gold_letter,
        })

    return tf_inputs, tf_labels, tf_rows



# Short Answer
def build_short_answer_inputs(
    rows: List[Dict[str, Any]],
    prompt_cot: str = "da",               
    system_msg: str = "You are a helpful assistant.",
) -> Tuple[List[List[dict]], List[str], List[Dict[str, Any]]]:

    sa_inputs: List[List[dict]] = []
    gold_answers: List[str] = []
    metas: List[Dict[str, Any]] = []

    for row in rows:
        question = (row.get("question") or "").strip()
        gold_letter = (row.get("answer") or "").strip().upper()
        if not question or not gold_letter:
            continue

        # 라벨→텍스트 매핑에서 정답 텍스트 찾기
        gold_text: Optional[str] = None
        for L in string.ascii_uppercase[:10]:  
            key = f"label_{L}"
            val = row.get(key)
            if L == gold_letter and isinstance(val, str) and val.strip():
                gold_text = val.strip()
                break
        if not gold_text:
            continue

        if prompt_cot == "cot":
            user = (
                "Answer the following question with the given context.\n\n"
                f"Question: {question}\n\n"
                "Let's think step by step.\n"
                "After reasoning, output the correct answer on the last line, between '<' and '>'. Output format example: <answer>\n\n"
            )

        else:
            user = (
                "Answer the following question with the given context.\n\n"
                f"Question: {question}\n\n"
                "Output only the correct answer between '<' and '>'. Do not add any extra text.\n"
                "Output format example: <answer>\n"
            )

        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user},
        ]
        sa_inputs.append(msgs)
        gold_answers.append(gold_text)
        metas.append({
            "question": question,
            "gold_answer_text": gold_text,
            "orig_gold_letter": gold_letter,
        })

    return sa_inputs, gold_answers, metas
