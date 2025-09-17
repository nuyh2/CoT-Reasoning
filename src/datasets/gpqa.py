# src/datasets/gpqa.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import random
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from src.utils import build_template_mcq, _make_labels

_GPQA_COLUMNS = [
    "Question", "Correct Answer",
    "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
    "Subdomain", "High-level domain",
]

class GPQADataset:
    name = "gpqa"

    def __init__(self, split: str = "train", config: str = "gpqa_main"):
        self.split = split
        self.config = config
        self._ds: Optional[Dataset] = None

    def load(self) -> Dataset:
        ds = load_dataset("Idavidrein/gpqa", self.config)
        ds = ds.select_columns(_GPQA_COLUMNS)[self.split]
        self._ds = ds
        return ds

    def build_inputs(
        self,
        limit: Optional[int] = None,
        prompt_cot: str = "",
        seed: int = 42,
        show_tqdm: bool = True,
        allow_think_tags: bool = True,
        include_rows: bool = False,   # ← 추가: 기존 outputs 스키마를 같이 생성/반환
    ) -> Tuple[
        List[List[dict]],            # input_list (chat messages)
        List[str],                   # answer_labels (A..D)
        Dataset,                     # raw dataset
        Optional[List[Dict[str, Any]]]  # rows(outputs 초기값) - include_rows=True일 때만
    ]:
        ds = self._ds or self.load()
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        rng = random.Random(seed)
        input_list: List[List[dict]] = []
        answer_labels: List[str] = []
        rows: List[Dict[str, Any]] = []

        it = range(len(ds))
        if show_tqdm:
            it = tqdm(it, desc="Building Prompts", total=len(ds))

        for i in it:
            q = ds[i]["Question"]
            choices = [
                ds[i]["Correct Answer"],
                ds[i]["Incorrect Answer 1"],
                ds[i]["Incorrect Answer 2"],
                ds[i]["Incorrect Answer 3"],
            ]
            rng.shuffle(choices)

            # 프롬프트(messages)
            msgs = build_template_mcq(
                question=q,
                choices=choices,
                prompt_cot=prompt_cot,
                allow_think_tags=allow_think_tags,
            )
            input_list.append(msgs)

            # 정답 라벨(A..D) 계산
            gold_text = ds[i]["Correct Answer"]
            labels = _make_labels(len(choices))  # ["A","B","C","D"]
            gold_label = labels[choices.index(gold_text)]
            answer_labels.append(gold_label)

            # (옵션) 기존 코드 호환용 outputs row 생성
            if include_rows:
                row = {
                    "question": q,
                    "label_A": choices[0],
                    "label_B": choices[1],
                    "label_C": choices[2],
                    "label_D": choices[3],
                    "answer": gold_label,  # ← 기존 코드 동일 키
                    "Subdomain": ds[i]["Subdomain"],
                    "High-level domain": ds[i]["High-level domain"],
                }
                rows.append(row)

        return input_list, answer_labels, ds, (rows if include_rows else None)
