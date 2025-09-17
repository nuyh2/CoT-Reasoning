# src/datasets/musr.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import random
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from src.utils import build_template_mcq, _make_labels
import ast

_COLUMNS = ["question", "narrative", "answer_index", "answer_choice", "choices"]

class MuSRMurderMysteriesBinaryDataset:
    
    name = "musr_murder_mysteries_binary"

    def __init__(self, split: str = "murder_mysteries", hf_path: str = "TAUR-Lab/MuSR"):
        self.split = split
        self.hf_path = hf_path
        self._ds: Optional[Dataset] = None

    def load(self) -> Dataset:
        ds = load_dataset(self.hf_path)
        ds = ds.select_columns(_COLUMNS)[self.split]
        self._ds = ds
        return ds

    def build_inputs(
        self,
        limit: Optional[int] = None,
        prompt_cot: str = "",
        seed: int = 42,
        show_tqdm: bool = True,
        allow_think_tags: bool = True,
        include_rows: bool = False,   # outputs 호환 rows 생성 여부
    ) -> Tuple[
        List[List[dict]],                 # input_list 
        List[str],                        # answer_labels 
        Dataset,                          # raw dataset
        Optional[List[Dict[str, Any]]]    
    ]:
        ds = self._ds or self.load()
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        rng = random.Random(seed)
        input_list: List[List[dict]] = []
        answer_labels: List[str] = []
        rows: List[Dict[str, Any]] = []

        iterator = range(len(ds))
        if show_tqdm:
            iterator = tqdm(iterator, desc="Building Prompts (MuSR)", total=len(ds))

        for i in iterator:
            ex = ds[i]
            q = ex["question"]
            narrative = ex["narrative"]
            choices: List[str] = ex["choices"]
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    continue
            ans_idx = int(ex["answer_index"])

            # 유효성 체크
            if not isinstance(choices, list) or len(choices) < 2:
                continue
            if ans_idx < 0 or ans_idx >= len(choices):
                continue

            correct_text = choices[ans_idx]

            if len(choices) > 10:
                distractors = [c for j, c in enumerate(choices) if j != ans_idx]
                sampled = rng.sample(distractors, 9)
                options = [correct_text] + sampled
            else:
                options = list(choices)

            options = [o for o in options if isinstance(o, str) and o.strip()]
            if len(options) < 2:
                continue
            rng.shuffle(options)

            # Question에 Context 포함
            q_with_ctx = f"Context:\n{narrative}\n\n{q}"

            msgs = build_template_mcq(
                question=q_with_ctx,
                choices=options,
                prompt_cot=prompt_cot,        
                allow_think_tags=allow_think_tags,
            )
            input_list.append(msgs)

            dyn_labels = _make_labels(len(options))
            gold_label = dyn_labels[options.index(correct_text)]
            answer_labels.append(gold_label)

            if include_rows:
                row: Dict[str, Any] = {
                    "question": q_with_ctx,
                    "answer": gold_label,
                }
                for idx_opt, L in enumerate(dyn_labels):
                    row[f"label_{L}"] = options[idx_opt]
                rows.append(row)

        return input_list, answer_labels, ds, (rows if include_rows else None)
