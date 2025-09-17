# src/datasets/arc.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import random
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from src.utils import build_template_mcq, _make_labels

_COLUMNS = ["question", "answerKey", "choices"]

class ARCDataset:

    name = "arc_challenge"

    def __init__(
        self,
        split: str = "validation",
        hf_path: str = "allenai/ai2_arc",
        subset: str = "ARC-Challenge",
    ):
        self.split = split
        self.hf_path = hf_path
        self.subset = subset
        self._ds: Optional[Dataset] = None

    def load(self) -> Dataset:
        ds = load_dataset(self.hf_path, self.subset)
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
        List[List[dict]],            # input_list (chat messages)
        List[str],                   # answer_labels (A..*)
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

        iterator = range(len(ds))
        if show_tqdm:
            iterator = tqdm(iterator, desc="Building Prompts", total=len(ds))

        for i in iterator:
            q = ds[i]["question"]
            answer_key = (ds[i]["answerKey"] or "").strip().upper()  

            # choices
            labels_in = ds[i]["choices"]["label"]   #
            texts_in  = ds[i]["choices"]["text"]    
            lab2txt = {str(lab).strip().upper(): txt for lab, txt in zip(labels_in, texts_in)}

            # 정답 텍스트 추출
            if answer_key not in lab2txt:
                continue
            correct_text = lab2txt[answer_key]

            # 선택지 구성
            options = [correct_text] + [lab2txt[lab] for lab in labels_in if str(lab).strip().upper() != answer_key]
            # 빈문자/None 제거
            options = [opt for opt in options if isinstance(opt, str) and opt.strip()]
            
            if not (2 <= len(options) <= 10):
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

            dyn_labels = _make_labels(len(options))  # ex) ['A','B','C','D']
            gold_label = dyn_labels[options.index(correct_text)]
            answer_labels.append(gold_label)

            # (옵션) 기존 코드 호환용 rows 생성
            if include_rows:
                row = {
                    "question": q,
                    "answer": gold_label,
                
                }
                for idx, L in enumerate(dyn_labels):
                    row[f"label_{L}"] = options[idx]
                rows.append(row)

        return input_list, answer_labels, ds, (rows if include_rows else None)
