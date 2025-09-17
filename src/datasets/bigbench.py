# src/datasets/bigbench_anu.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import random

from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from src.utils import build_template_mcq, _make_labels


class BigBenchDataset:

    name = "bigbench_anu_4choice"

    def __init__(
        self,
        split: str = "validation",
        path: str = "tasksource/bigbench",
        subset: str = "abstract_narrative_understanding",
    ):
        self.split = split
        self.path = path
        self.subset = subset
        self._ds: Optional[Dataset] = None

        # 통계
        self.total = 0
        self.used = 0
        self.skipped_too_few = 0
        self.skipped_bad_scores = 0

    def load(self) -> Dataset:
        ds = load_dataset(self.path, self.subset, split=self.split)
        self._ds = ds
        return ds

    def build_inputs(
        self,
        limit: Optional[int] = None,
        prompt_cot: str = "",
        seed: int = 42,
        show_tqdm: bool = True,
        verbose: bool = True,
        allow_think_tags: bool = True,
        include_rows: bool = False,   
    ) -> Tuple[
        List[List[dict]],            # input_list (chat messages)
        List[str],                   # answer_labels (A..*)
        Dataset,                     # raw dataset
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
            iterator = tqdm(iterator, desc="ANU->4-choice", total=len(ds))

        self.total = len(ds)
        self.used = 0
        self.skipped_too_few = 0
        self.skipped_bad_scores = 0

        for i in iterator:
            ex = ds[i]
            question: str = ex.get("inputs", "")
            mct: List[str] = ex.get("multiple_choice_targets", [])
            mcs: List[float] = ex.get("multiple_choice_scores", [])
            idx = ex.get("idx")

            if not isinstance(mct, list) or not isinstance(mcs, list):
                self.skipped_bad_scores += 1
                continue
            if len(mct) < 4 or len(mct) != len(mcs):
                self.skipped_too_few += 1
                continue

            # 정답 인덱스
            try:
                scores = [float(x) for x in mcs]
            except Exception:
                self.skipped_bad_scores += 1
                continue
            correct_idx = max(range(len(scores)), key=lambda k: (scores[k], -k))
            correct_text = mct[correct_idx]

            # 정답 + 무작위 오답 3개 → 4지
            distractors = [mct[j] for j in range(len(mct)) if j != correct_idx]
            try:
                sampled = rng.sample(distractors, 3)
            except ValueError:
                self.skipped_too_few += 1
                continue

            options = [correct_text] + sampled
            # 빈/None 제거
            options = [o for o in options if isinstance(o, str) and o.strip()]
            if len(options) != 4:
                self.skipped_too_few += 1
                continue

            rng.shuffle(options)

            # 프롬프트(messages) — 통일된 MCQ 빌더 사용
            msgs = build_template_mcq(
                question=question,
                choices=options,
                prompt_cot=prompt_cot,       
                allow_think_tags=allow_think_tags,
            )
            input_list.append(msgs)

            # 정답 라벨(A..D)
            dyn_labels = _make_labels(len(options))  
            gold_label = dyn_labels[options.index(correct_text)]
            answer_labels.append(gold_label)

            # rows 
            if include_rows:
                row: Dict[str, Any] = {
                    "question": question,
                    "answer": gold_label,
                    "subset": self.subset,
                    "idx": idx,
                    "orig_num_choices": len(mct),
                }
                for idx_opt, L in enumerate(dyn_labels):
                    row[f"label_{L}"] = options[idx_opt]
                rows.append(row)

            self.used += 1

        if verbose:
            skipped = self.total - self.used
            print("\n[ANU->4c] summary")
            print(f"  total   : {self.total}")
            print(f"  used    : {self.used}")
            print(f"  skipped : {skipped}")
            print(f"    - too few / len mismatch : {self.skipped_too_few}")
            print(f"    - bad scores             : {self.skipped_bad_scores}")

        return input_list, answer_labels, ds, (rows if include_rows else None)
