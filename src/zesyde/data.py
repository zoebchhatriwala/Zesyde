"""Data preparation and freezing for Zesyde benchmarks.

Transforms source datasets (AG News, DAIR-AI Emotion, SST-5) into frozen
`DecisionExample` sets conforming to M1 exit criteria (>= 300 held-out cases per workflow).
"""
import json
from pathlib import Path
from typing import List, Generator
from datasets import load_dataset
from zesyde.schema import DecisionExample, QuestionType

AG_NEWS_OPTIONS = ['World', 'Sports', 'Business', 'Sci/Tech']
EMOTION_OPTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
SST5_OPTIONS = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

def build_ag_news_eval(limit: int = 400) -> List[DecisionExample]:
    ds = load_dataset('SetFit/ag_news', split='test')
    examples = []
    for idx, row in enumerate(ds):
        if len(examples) >= limit:
            break
        examples.append(
            DecisionExample(
                id=f"ag_news_eval_{idx:04d}",
                context=row['text'],
                question="What category best describes this news article?",
                question_type=QuestionType.CHOICE,
                options=AG_NEWS_OPTIONS,
                target=int(row['label']),
                metadata={'dataset': 'ag_news', 'split': 'test'}
            )
        )
    return examples

def build_emotion_eval(limit: int = 400) -> List[DecisionExample]:
    ds = load_dataset('dair-ai/emotion', 'split', split='test')
    examples = []
    for idx, row in enumerate(ds):
        if len(examples) >= limit:
            break
        examples.append(
            DecisionExample(
                id=f"emotion_eval_{idx:04d}",
                context=row['text'],
                question="What emotion is primarily conveyed by this text?",
                question_type=QuestionType.CHOICE,
                options=EMOTION_OPTIONS,
                target=int(row['label']),
                metadata={'dataset': 'dair_emotion', 'split': 'test'}
            )
        )
    return examples

def build_emotion_noul_eval(limit: int = 400) -> List[DecisionExample]:
    ds = load_dataset('dair-ai/emotion', 'split', split='test')
    examples = []
    for idx, row in enumerate(ds):
        if len(examples) >= limit:
            break
        is_joy = int(row['label']) == 1
        examples.append(
            DecisionExample(
                id=f"emotion_noul_eval_{idx:04d}",
                context=row['text'],
                question="Does this statement convey joy or happiness?",
                question_type=QuestionType.NOUL,
                options=['No', 'Yes'],
                target=1 if is_joy else 0,
                metadata={'dataset': 'dair_emotion_noul', 'target_concept': 'joy'}
            )
        )
    return examples

def build_sst5_score_eval(limit: int = 400) -> List[DecisionExample]:
    ds = load_dataset('SetFit/sst5', split='test')
    examples = []
    for idx, row in enumerate(ds):
        if len(examples) >= limit:
            break
        examples.append(
            DecisionExample(
                id=f"sst5_score_eval_{idx:04d}",
                context=row['text'],
                question="Rate the sentiment of this review from 1 to 5.",
                question_type=QuestionType.SCORE,
                options=SST5_OPTIONS,
                target=int(row['label']),
                metadata={'dataset': 'sst5', 'scale': '1-5'}
            )
        )
    return examples

def freeze_dataset(examples: List[DecisionExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(ex.model_dump_json() + '\n')

def load_frozen_dataset(path: Path) -> List[DecisionExample]:
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(DecisionExample.model_validate_json(line))
    return examples
