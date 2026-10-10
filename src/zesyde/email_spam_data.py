"""Multilingual Real-World Email & Spam/Phishing Dataset Loader.

Loads real, open-source email and message classification datasets from Hugging Face:
1. `zefang-liu/phishing-email-dataset`: Real safe transactional/business emails vs phishing lures.
2. `SetFit/enron_spam`: Real corporate emails with formal headers, dates, and numbers.
3. `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`: Real multilingual communication across
   11 languages (Spanish, Hindi, Arabic, German, French, Chinese, Russian, Bengali, Urdu, English).
"""

import random
from typing import List, Optional
from datasets import load_dataset
from zesyde.schema import DecisionExample, QuestionType


QUESTION_STYLES = [
    ("What is the category of this email?", ["Legitimate", "Spam"]),
    ("What is the category of this email?", ["Spam", "Legitimate"]),
    ("What is the classification of this email?", ["Legitimate", "Spam"]),
    ("What is the classification of this email?", ["Spam", "Legitimate"]),
    ("What is the security classification of this message?", ["Legitimate", "Spam / Fraud"]),
    ("What is the security classification of this message?", ["Spam / Fraud", "Legitimate"]),
    ("Is this email legitimate or spam?", ["Legitimate", "Spam"]),
    ("Is this email legitimate or spam?", ["Spam", "Legitimate"]),
]


def load_real_spam_ham_dataset(limit: int = 2000, seed: int = 42) -> List[DecisionExample]:
    """Loads a balanced, real-world multilingual email spam & ham dataset from Hugging Face."""
    rng = random.Random(seed)
    examples: List[DecisionExample] = []

    target_per_source = limit // 3
    
    # -------------------------------------------------------------------------
    # 1. Real Phishing vs. Safe Emails (zefang-liu/phishing-email-dataset)
    # -------------------------------------------------------------------------
    try:
        ds_phish = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        safe_emails = [r["Email Text"] for r in ds_phish if r.get("Email Type") == "Safe Email" and r.get("Email Text")]
        phish_emails = [r["Email Text"] for r in ds_phish if r.get("Email Type") == "Phishing Email" and r.get("Email Text")]
        rng.shuffle(safe_emails)
        rng.shuffle(phish_emails)

        count_phish = target_per_source // 2
        for idx in range(min(count_phish, len(safe_emails))):
            text = safe_emails[idx].strip()
            if len(text) < 20:
                continue
            q_text, opts = rng.choice(QUESTION_STYLES)
            target = opts.index("Legitimate") if "Legitimate" in opts else 0
            examples.append(DecisionExample(
                id=f"real_safe_{idx:05d}",
                context=text[:2500],
                question=q_text,
                question_type=QuestionType.CHOICE,
                options=opts,
                target=target,
                metadata={"dataset": "zefang_liu_phishing", "lang": "en", "is_spam": False}
            ))

        for idx in range(min(count_phish, len(phish_emails))):
            text = phish_emails[idx].strip()
            if len(text) < 20:
                continue
            q_text, opts = rng.choice(QUESTION_STYLES)
            target = 1 if opts[0] == "Legitimate" else 0
            examples.append(DecisionExample(
                id=f"real_phish_{idx:05d}",
                context=text[:2500],
                question=q_text,
                question_type=QuestionType.CHOICE,
                options=opts,
                target=target,
                metadata={"dataset": "zefang_liu_phishing", "lang": "en", "is_spam": True}
            ))
    except Exception as e:
        print(f"Warning loading zefang-liu/phishing-email-dataset: {e}")

    # -------------------------------------------------------------------------
    # 2. Real Enron Corporate Emails (SetFit/enron_spam)
    # -------------------------------------------------------------------------
    try:
        ds_enron = load_dataset("SetFit/enron_spam", split="train")
        enron_ham = [r["text"] for r in ds_enron if r.get("label_text") == "ham" and r.get("text")]
        enron_spam = [r["text"] for r in ds_enron if r.get("label_text") == "spam" and r.get("text")]
        rng.shuffle(enron_ham)
        rng.shuffle(enron_spam)

        count_enron = target_per_source // 2
        for idx in range(min(count_enron, len(enron_ham))):
            text = enron_ham[idx].strip()
            if len(text) < 20:
                continue
            q_text, opts = rng.choice(QUESTION_STYLES)
            target = opts.index("Legitimate") if "Legitimate" in opts else 0
            examples.append(DecisionExample(
                id=f"real_enron_ham_{idx:05d}",
                context=text[:2500],
                question=q_text,
                question_type=QuestionType.CHOICE,
                options=opts,
                target=target,
                metadata={"dataset": "enron_spam", "lang": "en", "is_spam": False}
            ))

        for idx in range(min(count_enron, len(enron_spam))):
            text = enron_spam[idx].strip()
            if len(text) < 20:
                continue
            q_text, opts = rng.choice(QUESTION_STYLES)
            target = 1 if opts[0] == "Legitimate" else 0
            examples.append(DecisionExample(
                id=f"real_enron_spam_{idx:05d}",
                context=text[:2500],
                question=q_text,
                question_type=QuestionType.CHOICE,
                options=opts,
                target=target,
                metadata={"dataset": "enron_spam", "lang": "en", "is_spam": True}
            ))
    except Exception as e:
        print(f"Warning loading SetFit/enron_spam: {e}")

    # -------------------------------------------------------------------------
    # 3. Real Multilingual Dataset across 11 Languages (dbarbedillo)
    # -------------------------------------------------------------------------
    try:
        ds_multi = load_dataset("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset", split="train")
        lang_cols = ["text_es", "text_hi", "text_ar", "text_fr", "text_de", "text_ru", "text_zh", "text_bn", "text_mr", "text_ur"]
        per_lang = max(10, target_per_source // len(lang_cols))

        for col in lang_cols:
            lang_code = col.replace("text_", "")
            c = 0
            shuffled = list(ds_multi)
            rng.shuffle(shuffled)
            for row in shuffled:
                if c >= per_lang:
                    break
                txt = row.get(col)
                if not txt or not txt.strip() or len(txt.strip()) < 10:
                    continue
                is_spam = (row.get("labels") == "spam")
                q_text, opts = rng.choice(QUESTION_STYLES)
                target = (1 if opts[0] == "Legitimate" else 0) if is_spam else (opts.index("Legitimate") if "Legitimate" in opts else 0)

                examples.append(DecisionExample(
                    id=f"real_multi_{lang_code}_{c:05d}",
                    context=txt.strip()[:1000],
                    question=q_text,
                    question_type=QuestionType.CHOICE,
                    options=opts,
                    target=target,
                    metadata={"dataset": "multilingual_spam", "lang": lang_code, "is_spam": is_spam}
                ))
                c += 1
    except Exception as e:
        print(f"Warning loading dbarbedillo multilingual spam dataset: {e}")

    rng.shuffle(examples)
    return examples[:limit]
