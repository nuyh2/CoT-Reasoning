# src/datasets/commonsense_qa.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import random
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from src.utils import build_template_mcq, _make_labels

_COLUMNS = ["question", "answerKey", "choices", "question_concept"]

class CommonsenseQADataset:

    name = "commonsense_qa"

    def __init__(self, split: str = "validation", hf_path: str = "tau/commonsense_qa"):
        self.split = split
        self.hf_path = hf_path
        self._ds: Optional[Dataset] = None

    def load(self) -> Dataset:
        ds = load_dataset(self.hf_path)  # splits: train/validation/test
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
        include_rows: bool = False,  
    ) -> Tuple[
        List[List[dict]],                 # input_list (chat messages)
        List[str],                        # answer_labels (A..*)
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
            iterator = tqdm(iterator, desc="CSQA->4-choice", total=len(ds))

        for i in iterator:
            ex = ds[i]
            q = ex["question"]
            answer_key = (ex["answerKey"] or "").strip().upper()  

            labels = [str(l).strip().upper() for l in ex["choices"]["label"]]
            texts  = ex["choices"]["text"]


            if answer_key not in labels or len(labels) != len(texts):
                continue

            lab2idx = {lab: idx for idx, lab in enumerate(labels)}
            correct_text = texts[lab2idx[answer_key]]

            # 4지 구성
            distractors = [texts[j] for j, lab in enumerate(labels) if lab != answer_key]
            distractors = [d for d in distractors if isinstance(d, str) and d.strip()]
            if len(distractors) < 3:
                continue
            sampled = rng.sample(distractors, 3)

            options = [correct_text] + sampled
            options = [opt for opt in options if isinstance(opt, str) and opt.strip()]
            if len(options) != 4:
                continue
            rng.shuffle(options)

            # 프롬프트(messages)
            msgs = build_template_mcq(
                question=q,
                choices=options,
                prompt_cot=prompt_cot,       
                allow_think_tags=allow_think_tags,
            )
            input_list.append(msgs)

            # 정답 라벨
            dyn_labels = _make_labels(len(options))  
            gold_label = dyn_labels[options.index(correct_text)]
            answer_labels.append(gold_label)

            if include_rows:
                row: Dict[str, Any] = {
                    "question": q,
                    "answer": gold_label,
                    "question_concept": ex.get("question_concept"),
                }
                for idx_opt, L in enumerate(dyn_labels):
                    row[f"label_{L}"] = options[idx_opt]
                rows.append(row)

        return input_list, answer_labels, ds, (rows if include_rows else None)
