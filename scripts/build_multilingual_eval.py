"""Builds frozen multilingual evaluation datasets in data/eval_multilingual/."""
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from zesyde.schema import DecisionExample, QuestionType
from zesyde.data import freeze_dataset

def build_multilingual_eval_suite(output_dir: Path, limit_per_set: int = 400):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Belebele Spanish (Choice - 4 options)
    print("Generating Belebele Spanish (spa_Latn)...")
    ds = load_dataset("facebook/belebele", "spa_Latn", split="test")
    exs_es = []
    for idx, row in enumerate(ds):
        if len(exs_es) >= limit_per_set: break
        exs_es.append(DecisionExample(
            id=f"belebele_es_{idx:04d}",
            context=row['flores_passage'],
            question=row['question'],
            question_type=QuestionType.CHOICE,
            options=[row['mc_answer1'], row['mc_answer2'], row['mc_answer3'], row['mc_answer4']],
            target=int(row['correct_answer_num']) - 1,
            metadata={'dataset': 'belebele', 'lang': 'es', 'split': 'test'}
        ))
    freeze_dataset(exs_es, output_dir / "belebele_choice_es.jsonl")

    # 2. Belebele Hindi (Choice - 4 options, Devanagari script)
    print("Generating Belebele Hindi (hin_Deva)...")
    ds = load_dataset("facebook/belebele", "hin_Deva", split="test")
    exs_hi = []
    for idx, row in enumerate(ds):
        if len(exs_hi) >= limit_per_set: break
        exs_hi.append(DecisionExample(
            id=f"belebele_hi_{idx:04d}",
            context=row['flores_passage'],
            question=row['question'],
            question_type=QuestionType.CHOICE,
            options=[row['mc_answer1'], row['mc_answer2'], row['mc_answer3'], row['mc_answer4']],
            target=int(row['correct_answer_num']) - 1,
            metadata={'dataset': 'belebele', 'lang': 'hi', 'split': 'test'}
        ))
    freeze_dataset(exs_hi, output_dir / "belebele_choice_hi.jsonl")

    # 3. Belebele Arabic (Choice - 4 options, Arabic script)
    print("Generating Belebele Arabic (arb_Arab)...")
    ds = load_dataset("facebook/belebele", "arb_Arab", split="test")
    exs_ar = []
    for idx, row in enumerate(ds):
        if len(exs_ar) >= limit_per_set: break
        exs_ar.append(DecisionExample(
            id=f"belebele_ar_{idx:04d}",
            context=row['flores_passage'],
            question=row['question'],
            question_type=QuestionType.CHOICE,
            options=[row['mc_answer1'], row['mc_answer2'], row['mc_answer3'], row['mc_answer4']],
            target=int(row['correct_answer_num']) - 1,
            metadata={'dataset': 'belebele', 'lang': 'ar', 'split': 'test'}
        ))
    freeze_dataset(exs_ar, output_dir / "belebele_choice_ar.jsonl")

    # 4. XNLI French (Choice - 3 options: Entailment, Neutral, Contradiction)
    print("Generating XNLI French (fr)...")
    ds = load_dataset("facebook/xnli", "fr", split="test")
    exs_fr = []
    nli_options = ["Entailment", "Neutral", "Contradiction"]
    for idx, row in enumerate(ds):
        if len(exs_fr) >= limit_per_set: break
        exs_fr.append(DecisionExample(
            id=f"xnli_choice_fr_{idx:04d}",
            context=row['premise'],
            question=f"Quelle est la relation logique avec: {row['hypothesis']}?",
            question_type=QuestionType.CHOICE,
            options=nli_options,
            target=int(row['label']),
            metadata={'dataset': 'xnli', 'lang': 'fr', 'split': 'test'}
        ))
    freeze_dataset(exs_fr, output_dir / "xnli_choice_fr.jsonl")

    # 5. XNLI German (Noul - 2 options: Nein, Ja / No, Yes)
    print("Generating XNLI German Noul (de)...")
    ds = load_dataset("facebook/xnli", "de", split="test")
    exs_de_noul = []
    for idx, row in enumerate(ds):
        if len(exs_de_noul) >= limit_per_set: break
        is_entailment = 1 if int(row['label']) == 0 else 0
        exs_de_noul.append(DecisionExample(
            id=f"xnli_noul_de_{idx:04d}",
            context=row['premise'],
            question=f"Folgt diese Aussage logisch daraus: {row['hypothesis']}?",
            question_type=QuestionType.NOUL,
            options=["Nein", "Ja"],
            target=is_entailment,
            metadata={'dataset': 'xnli_noul', 'lang': 'de', 'split': 'test'}
        ))
    freeze_dataset(exs_de_noul, output_dir / "xnli_noul_de.jsonl")

    print(f"\nMultilingual evaluation suite generated in {output_dir}")

if __name__ == "__main__":
    build_multilingual_eval_suite(Path("data/eval_multilingual"), limit_per_set=400)
