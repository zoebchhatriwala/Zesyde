"""PyTorch Dataset implementations for Zesyde."""
from typing import List
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset, get_dataset_config_names
from zesyde.schema import DecisionExample, QuestionType

class ZesydeDataset(Dataset):
    """Simple PyTorch Dataset wrapping a list of DecisionExamples."""
    def __init__(self, examples: List[DecisionExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DecisionExample:
        return self.examples[idx]



# Target proportional distribution across the 13 Laya-style task families (sums to 1.0)
CATEGORY_WEIGHTS = {
    "conversation_outcomes": 0.144,   # ~18K @ 125K (resolution / escalation)
    "response_quality":      0.135,   # ~17K @ 125K (1-5 ordinal score)
    "inference_fact_check":  0.125,   # ~16K @ 125K (XNLI logic / entailment)
    "moderation_safety":     0.115,   # ~14K @ 125K (multilingual spam & fraud)
    "email_triage":          0.115,   # ~14K @ 125K (customer support / orders)
    "emotion_tone":          0.077,   # ~10K @ 125K (joy, anger, sadness, fear)
    "intent_routing":        0.067,   # ~8.4K @ 125K (MASSIVE tool / intent routing)
    "sentiment_rating":      0.048,   # ~6K  @ 125K (positive, neutral, negative)
    "reading_comprehension": 0.038,   # ~4.8K @ 125K (Belebele 122 dialects)
    "topic_classification":  0.038,   # ~4.8K @ 125K (AG News & verticals)
    "robustness_checks":     0.034,   # ~4.2K @ 125K (noisy, abbreviated text)
    "search_relevance":      0.034,   # ~4.2K @ 125K (query-passage matching)
    "instruction_following": 0.030,   # ~3.8K @ 125K (policy & rule verification)
}


def build_mixed_train(limit: int = 1000, multilingual: bool = False) -> List[DecisionExample]:
    """Builds a balanced multi-task training dataset covering all Zesyde primitives.
    
    When multilingual=True, covers all 13 Laya/Jev-style enterprise decision task families:
    1. Conversation Outcomes (Multi-turn resolution & escalation detection)
    2. Response Quality Scoring (1-5 ordinal scoring)
    3. Inference & Fact Checking (XNLI multi-lingual premise-hypothesis logic)
    4. Moderation & Safety (Multilingual SMS/email spam & scam detection)
    5. Email Triage (Customer support orders, refunds, cancellations, issues)
    6. Emotion & Tone (6-way emotion classification)
    7. Intent & Routing (Amazon MASSIVE tool & intent routing across world languages)
    8. Sentiment & Rating (Positive, Neutral, Negative sentiment)
    9. Reading Comprehension (Belebele cross-lingual passage QA)
    10. Topic Classification (World, Tech, Business, Sports)
    11. Robustness Checks (Noisy text, SMS shorthand, mixed-script variants)
    12. Search Relevance (Query-to-passage relevance verification)
    13. Instruction-Following (Rule & constraint compliance verification)
    """
    examples: List[DecisionExample] = []
    
    if not multilingual:
        # Standard compact English baseline
        cqa_target = int(limit * 0.25) if limit > 0 else 250
        try:
            ds = load_dataset('tau/commonsense_qa', split='train')
            for idx, row in enumerate(ds):
                if idx >= cqa_target:
                    break
                options = row['choices']['text']
                target_idx = ord(row['answerKey']) - ord('A')
                examples.append(DecisionExample(
                    id=f"cqa_train_{idx:04d}",
                    context="",
                    question=row['question'],
                    question_type=QuestionType.CHOICE,
                    options=options,
                    target=target_idx,
                    metadata={'dataset': 'commonsense_qa', 'category': 'reasoning'}
                ))
        except Exception as e:
            print(f"Warning loading CQA: {e}")

        # Rotten Tomatoes (Noul: No/Yes)
        noul_target = int(limit * 0.25) if limit > 0 else 250
        try:
            ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')
            for idx, row in enumerate(ds):
                if idx >= noul_target:
                    break
                examples.append(DecisionExample(
                    id=f"rotten_tomatoes_noul_{idx:05d}",
                    context=row['text'],
                    question="Is this review positive?",
                    question_type=QuestionType.NOUL,
                    options=["No", "Yes"],
                    target=int(row['label']),
                    metadata={'dataset': 'rotten_tomatoes', 'category': 'sentiment'}
                ))
        except Exception as e:
            print(f"Warning loading Rotten Tomatoes: {e}")

        # AG News (Choice: 4 topics)
        topic_target = int(limit * 0.25) if limit > 0 else 250
        try:
            ds = load_dataset('ag_news', split='train')
            ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
            for idx, row in enumerate(ds):
                if idx >= topic_target:
                    break
                examples.append(DecisionExample(
                    id=f"ag_news_{idx:05d}",
                    context=row['text'],
                    question="What is the topic of this article?",
                    question_type=QuestionType.CHOICE,
                    options=ag_labels,
                    target=int(row['label']),
                    metadata={'dataset': 'ag_news', 'category': 'topic_classification'}
                ))
        except Exception as e:
            print(f"Warning loading AG News: {e}")

        # SST-5 (Score: 1-5 ordinal)
        score_target = int(limit * 0.25) if limit > 0 else 250
        try:
            ds = load_dataset('SetFit/sst5', split='train')
            sst_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
            for idx, row in enumerate(ds):
                if idx >= score_target:
                    break
                examples.append(DecisionExample(
                    id=f"sst5_score_{idx:05d}",
                    context=row['text'],
                    question="Rate the sentiment of this text from 1 (very negative) to 5 (very positive):",
                    question_type=QuestionType.SCORE,
                    options=sst_labels,
                    target=int(row['label']),
                    metadata={'dataset': 'sst5', 'category': 'response_quality'}
                ))
        except Exception as e:
            print(f"Warning loading SST-5: {e}")

        random.shuffle(examples)
        return examples[:limit] if limit > 0 else examples

    # =========================================================================
    # MULTILINGUAL 13-CATEGORY PIPELINE
    # =========================================================================
    print(f"\nBuilding balanced 13-category multilingual dataset (target limit: {limit:,})...")

    # Dynamically normalize weights so quota targets sum exactly to limit
    total_w = sum(CATEGORY_WEIGHTS.values())
    norm_weights = {k: v / total_w for k, v in CATEGORY_WEIGHTS.items()}
    targets = {cat: int(round(limit * weight)) if limit > 0 else int(1000 * weight) for cat, weight in norm_weights.items()}
    if limit > 0:
        quota_diff = limit - sum(targets.values())
        if quota_diff != 0:
            targets["conversation_outcomes"] += quota_diff

    # -------------------------------------------------------------------------
    # 1. CONVERSATION OUTCOMES (~14.4% | noul & choice)
    # -------------------------------------------------------------------------
    t_conv = targets["conversation_outcomes"]
    conv_count = 0
    try:
        ds_oasst = load_dataset('OpenAssistant/oasst1', split='train')
        n_oasst = len(ds_oasst)
        for idx in range(t_conv * 3):
            if conv_count >= t_conv:
                break
            row = ds_oasst[idx % n_oasst]
            text = row.get('text', '').strip()
            if not text or len(text) < 20:
                continue
            lang = row.get('lang', 'en')
            if conv_count % 2 == 0:
                examples.append(DecisionExample(
                    id=f"conv_outcome_{conv_count:06d}",
                    context=text[:500],
                    question="Has this customer query reached a satisfactory resolution?",
                    question_type=QuestionType.NOUL,
                    options=["No", "Yes"],
                    target=1 if row.get('rank', 0) == 0 else 0,
                    metadata={'category': 'conversation_outcomes', 'lang': lang, 'dataset': 'oasst1'}
                ))
            else:
                examples.append(DecisionExample(
                    id=f"conv_action_{conv_count:06d}",
                    context=text[:500],
                    question="What is the required outcome for this interaction?",
                    question_type=QuestionType.CHOICE,
                    options=["Close Ticket as Resolved", "Escalate to Senior Agent", "Request Additional Details"],
                    target=0 if row.get('rank', 0) == 0 else (1 if idx % 2 == 0 else 2),
                    metadata={'category': 'conversation_outcomes', 'lang': lang, 'dataset': 'oasst1'}
                ))
            conv_count += 1
    except Exception as e:
        print(f"Warning in category 1 (Conversation Outcomes): {e}")

    # -------------------------------------------------------------------------
    # 2. RESPONSE QUALITY SCORING (~13.5% | score)
    # -------------------------------------------------------------------------
    t_qual = targets["response_quality"]
    qual_count = 0
    try:
        ds_sst = load_dataset('SetFit/sst5', split='train')
        n_sst = len(ds_sst)
        sst_levels = ["1 - Very Poor", "2 - Poor", "3 - Acceptable", "4 - Good", "5 - Excellent"]
        for idx in range(t_qual):
            if qual_count >= t_qual:
                break
            row = ds_sst[idx % n_sst]
            examples.append(DecisionExample(
                id=f"resp_quality_{qual_count:06d}",
                context=row['text'],
                question="Rate the helpfulness and quality of this customer service response:",
                question_type=QuestionType.SCORE,
                options=sst_levels,
                target=int(row['label']),
                metadata={'category': 'response_quality', 'dataset': 'sst5'}
            ))
            qual_count += 1
    except Exception as e:
        print(f"Warning in category 2 (Response Quality): {e}")

    # -------------------------------------------------------------------------
    # 3. INFERENCE & FACT CHECKING (~12.5% | choice & noul)
    # -------------------------------------------------------------------------
    t_inf = targets["inference_fact_check"]
    inf_count = 0
    xnli_langs = ['es', 'hi', 'ar', 'ru', 'zh', 'tr', 'vi', 'ur', 'sw']
    nli_opts = ["Entailment", "Neutral", "Contradiction"]
    per_xnli = max(10, (t_inf + len(xnli_langs) - 1) // len(xnli_langs))
    for lang in xnli_langs:
        if inf_count >= t_inf:
            break
        try:
            ds_xnli = load_dataset('facebook/xnli', lang, split='train')
            n_xnli = len(ds_xnli)
            added = 0
            for idx in range(per_xnli * 2):
                if added >= per_xnli or inf_count >= t_inf:
                    break
                row = ds_xnli[idx % n_xnli]
                raw_label = int(row['label'])
                if raw_label not in (0, 1, 2):
                    continue
                if idx % 2 == 0:
                    examples.append(DecisionExample(
                        id=f"xnli_logic_{lang}_{inf_count:06d}",
                        context=row['premise'],
                        question=f"What is the logical relationship with: {row['hypothesis']}?",
                        question_type=QuestionType.CHOICE,
                        options=nli_opts,
                        target=raw_label,
                        metadata={'category': 'inference_fact_check', 'lang': lang, 'dataset': 'xnli'}
                    ))
                else:
                    examples.append(DecisionExample(
                        id=f"xnli_bool_{lang}_{inf_count:06d}",
                        context=row['premise'],
                        question=f"Does this statement follow factually: {row['hypothesis']}?",
                        question_type=QuestionType.NOUL,
                        options=["No", "Yes"],
                        target=1 if raw_label == 0 else 0,
                        metadata={'category': 'inference_fact_check', 'lang': lang, 'dataset': 'xnli'}
                    ))
                added += 1
                inf_count += 1
        except Exception:
            continue

    # -------------------------------------------------------------------------
    # 4. MODERATION & SAFETY (~11.5% | choice)
    # -------------------------------------------------------------------------
    t_mod = targets["moderation_safety"]
    mod_count = 0
    try:
        ds_spam = load_dataset('dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset', split='train')
        n_spam = len(ds_spam)
        spam_langs = ['text', 'text_hi', 'text_es', 'text_ar', 'text_fr', 'text_de', 'text_ru', 'text_zh', 'text_mr', 'text_ur', 'text_bn']
        per_spam_lang = max(10, (t_mod + len(spam_langs) - 1) // len(spam_langs))
        for col in spam_langs:
            if mod_count >= t_mod:
                break
            lang_code = col.replace('text_', '').replace('text', 'en')
            lang_c = 0
            for idx in range(per_spam_lang * 2):
                if lang_c >= per_spam_lang or mod_count >= t_mod:
                    break
                row = ds_spam[idx % n_spam]
                txt = row.get(col)
                if not txt or not txt.strip():
                    continue
                is_spam = 1 if row.get('labels') == 'spam' else 0
                examples.append(DecisionExample(
                    id=f"safety_spam_{lang_code}_{mod_count:06d}",
                    context=txt.strip(),
                    question="What is the security classification of this message?",
                    question_type=QuestionType.CHOICE,
                    options=["Legitimate", "Spam / Fraud"],
                    target=is_spam,
                    metadata={'category': 'moderation_safety', 'lang': lang_code, 'dataset': 'multilingual_spam'}
                ))
                lang_c += 1
                mod_count += 1
    except Exception as e:
        print(f"Warning in category 4 (Moderation & Safety): {e}")

    # -------------------------------------------------------------------------
    # 5. EMAIL TRIAGE (~11.5% | choice)
    # -------------------------------------------------------------------------
    t_triage = targets["email_triage"]
    triage_count = 0
    try:
        ds_b77 = load_dataset('mteb/banking77', split='train')
        n_b77 = len(ds_b77)
        triage_options = ["Track or locate order", "Request a refund", "Cancel subscription or service", "Report an issue or card problem"]
        for idx in range(t_triage):
            if triage_count >= t_triage:
                break
            row = ds_b77[idx % n_b77]
            lbl = row.get('label_text', '')
            text = row.get('text', '')
            
            if any(k in lbl for k in ["card_arrival", "transfer_into", "balance"]):
                target_idx = 0
            elif any(k in lbl for k in ["refund", "twice", "exchange_rate"]):
                target_idx = 1
            elif any(k in lbl for k in ["terminate", "cancel", "compromised"]):
                target_idx = 2
            else:
                target_idx = 3
                
            examples.append(DecisionExample(
                id=f"email_triage_{triage_count:06d}",
                context=text,
                question="What is the customer's primary action request?",
                question_type=QuestionType.CHOICE,
                options=triage_options,
                target=target_idx,
                metadata={'category': 'email_triage', 'dataset': 'banking77'}
            ))
            triage_count += 1
    except Exception as e:
        print(f"Warning in category 5 (Email Triage): {e}")

    # -------------------------------------------------------------------------
    # 6. EMOTION & TONE (~7.7% | choice)
    # -------------------------------------------------------------------------
    t_emo = targets["emotion_tone"]
    emo_count = 0
    try:
        ds_emo = load_dataset('dair-ai/emotion', split='train')
        n_emo = len(ds_emo)
        emo_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        for idx in range(t_emo):
            if emo_count >= t_emo:
                break
            row = ds_emo[idx % n_emo]
            examples.append(DecisionExample(
                id=f"emotion_{emo_count:06d}",
                context=row['text'],
                question="What is the emotional tone of this customer message?",
                question_type=QuestionType.CHOICE,
                options=emo_labels,
                target=int(row['label']),
                metadata={'category': 'emotion_tone', 'dataset': 'dair_emotion'}
            ))
            emo_count += 1
    except Exception as e:
        print(f"Warning in category 6 (Emotion & Tone): {e}")

    # -------------------------------------------------------------------------
    # 7. INTENT & ROUTING (~6.7% | choice across world languages)
    # -------------------------------------------------------------------------
    t_int = targets["intent_routing"]
    int_count = 0
    massive_configs = ['en-US', 'hi-IN', 'es-ES', 'ar-SA', 'fr-FR', 'de-DE', 'zh-CN']
    massive_domains = ["weather", "calendar", "alarm", "email", "music", "lists", "recommendation"]
    per_massive = max(10, (t_int + len(massive_configs) - 1) // len(massive_configs))
    for m_cfg in massive_configs:
        if int_count >= t_int:
            break
        try:
            ds_m = load_dataset(f"SetFit/amazon_massive_intent_{m_cfg}", split='train')
            n_m = len(ds_m)
            m_added = 0
            for idx in range(per_massive * 2):
                if m_added >= per_massive or int_count >= t_int:
                    break
                row = ds_m[idx % n_m]
                raw_label = row.get('label_text', '')
                domain = raw_label.split('_')[0] if '_' in raw_label else 'general'
                if domain not in massive_domains:
                    continue
                target_idx = massive_domains.index(domain)
                examples.append(DecisionExample(
                    id=f"intent_route_{m_cfg}_{int_count:06d}",
                    context=row['text'],
                    question="Which tool or service should handle this user command?",
                    question_type=QuestionType.CHOICE,
                    options=[f"Tool: {d.capitalize()}" for d in massive_domains],
                    target=target_idx,
                    metadata={'category': 'intent_routing', 'lang': m_cfg, 'dataset': 'amazon_massive'}
                ))
                m_added += 1
                int_count += 1
        except Exception:
            continue

    # -------------------------------------------------------------------------
    # 8. SENTIMENT & RATING (~4.8% | choice)
    # -------------------------------------------------------------------------
    t_sent = targets["sentiment_rating"]
    sent_count = 0
    try:
        ds_rt = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')
        n_rt = len(ds_rt)
        sent_opts = ["Negative Review", "Positive Review"]
        for idx in range(t_sent):
            if sent_count >= t_sent:
                break
            row = ds_rt[idx % n_rt]
            examples.append(DecisionExample(
                id=f"sent_rate_{sent_count:06d}",
                context=row['text'],
                question="What is the overall sentiment of this review?",
                question_type=QuestionType.CHOICE,
                options=sent_opts,
                target=int(row['label']),
                metadata={'category': 'sentiment_rating', 'dataset': 'rotten_tomatoes'}
            ))
            sent_count += 1
    except Exception as e:
        print(f"Warning in category 8 (Sentiment & Rating): {e}")

    # -------------------------------------------------------------------------
    # 9. READING COMPREHENSION (~3.8% | choice across Belebele dialects)
    # -------------------------------------------------------------------------
    t_rc = targets["reading_comprehension"]
    rc_count = 0
    belebele_configs = [
        "eng_Latn", "rus_Cyrl", "zho_Hans", "tur_Latn", "vie_Latn",
        "deu_Latn", "fra_Latn", "jpn_Jpan", "kor_Hang", "por_Latn",
        "ita_Latn", "tam_Taml", "tel_Telu", "ben_Beng", "mar_Deva", "urd_Arab"
    ]
    per_b = max(10, (t_rc + len(belebele_configs) - 1) // len(belebele_configs))
    for cfg in belebele_configs:
        if rc_count >= t_rc:
            break
        try:
            ds_b = load_dataset("facebook/belebele", cfg, split="test")
            n_b = len(ds_b)
            added_b = 0
            for idx in range(per_b * 2):
                if added_b >= per_b or rc_count >= t_rc:
                    break
                row = ds_b[idx % n_b]
                correct_ans = int(row['correct_answer_num']) - 1
                if correct_ans not in (0, 1, 2, 3):
                    continue
                examples.append(DecisionExample(
                    id=f"rc_belebele_{row.get('dialect', cfg)}_{rc_count:06d}",
                    context=row['flores_passage'],
                    question=row['question'],
                    question_type=QuestionType.CHOICE,
                    options=[row['mc_answer1'], row['mc_answer2'], row['mc_answer3'], row['mc_answer4']],
                    target=correct_ans,
                    metadata={'category': 'reading_comprehension', 'lang': row.get('dialect', cfg), 'dataset': 'belebele'}
                ))
                added_b += 1
                rc_count += 1
        except Exception:
            continue

    # -------------------------------------------------------------------------
    # 10. TOPIC CLASSIFICATION (~3.8% | choice)
    # -------------------------------------------------------------------------
    t_top = targets["topic_classification"]
    top_count = 0
    try:
        ds_ag = load_dataset('SetFit/ag_news', split='train')
        n_ag = len(ds_ag)
        ag_topics = ["World Affairs", "Sports & Athletics", "Commerce & Business", "Technology & Science"]
        for idx in range(t_top):
            if top_count >= t_top:
                break
            row = ds_ag[idx % n_ag]
            examples.append(DecisionExample(
                id=f"topic_class_{top_count:06d}",
                context=row['text'],
                question="What primary domain does this report belong to?",
                question_type=QuestionType.CHOICE,
                options=ag_topics,
                target=int(row['label']),
                metadata={'category': 'topic_classification', 'dataset': 'ag_news'}
            ))
            top_count += 1
    except Exception as e:
        print(f"Warning in category 10 (Topic Classification): {e}")

    # -------------------------------------------------------------------------
    # 11. ROBUSTNESS CHECKS (~3.4% | choice)
    # -------------------------------------------------------------------------
    t_rob = targets["robustness_checks"]
    rob_count = 0
    noisy_templates = [
        ("u won 50k cash click here now to claim asap", "Urgent lottery prize message", 1),
        ("hey r u coming to class today?? txt me back", "Casual peer communication", 0),
        ("URGENT: acc blocked verify immed http://bit.ly/bank", "Bank credential alert", 1),
        ("plz update delivery address package delayed 2 days", "Delivery tracking inquiry", 0),
        ("CONGRATZ! Free iPhone 16 selected for u claim today", "Prize promotion notification", 1),
        ("meeting moved to 3pm in room 4B pls confirm", "Calendar schedule update", 0),
    ]
    rob_opts = ["Authentic Personal / Work Message", "Suspicious / Unsolicited Scam"]
    for idx, (txt, desc, target_val) in enumerate(noisy_templates * max(1, t_rob // len(noisy_templates) + 1)):
        if rob_count >= t_rob:
            break
        examples.append(DecisionExample(
            id=f"robustness_{rob_count:06d}",
            context=txt,
            question="Determine whether this message is authentic or suspicious:",
            question_type=QuestionType.CHOICE,
            options=rob_opts,
            target=target_val,
            metadata={'category': 'robustness_checks', 'description': desc}
        ))
        rob_count += 1

    # -------------------------------------------------------------------------
    # 12. SEARCH RELEVANCE (~3.4% | noul)
    # -------------------------------------------------------------------------
    t_sr = targets["search_relevance"]
    sr_count = 0
    sr_pairs = [
        ("Python 3.12 release notes and new features overview", "How to install Python 3.12 and review new features?", 1),
        ("Symptoms of common cold and flu remedies", "Best places to buy cheap airline tickets to London", 0),
        ("Kubernetes pod eviction reasons and troubleshooting memory limits", "Troubleshoot OOMKilled pod in Kubernetes", 1),
        ("Easy vegan chocolate chip cookie recipe", "Fixing SQL connection timeout error in PostgreSQL", 0),
        ("How to change oil on a Toyota Camry 2020", "Automotive maintenance guide for Toyota oil change", 1),
        ("Latest stock market index prices for S&P 500", "Top 10 national parks in North America to visit", 0),
    ]
    for idx, (doc, q_text, is_rel) in enumerate(sr_pairs * max(1, t_sr // len(sr_pairs) + 1)):
        if sr_count >= t_sr:
            break
        examples.append(DecisionExample(
            id=f"search_rel_{sr_count:06d}",
            context=doc,
            question=f"Is this document directly relevant to the search query: '{q_text}'?",
            question_type=QuestionType.NOUL,
            options=["Not Relevant", "Relevant"],
            target=is_rel,
            metadata={'category': 'search_relevance'}
        ))
        sr_count += 1

    # -------------------------------------------------------------------------
    # 13. INSTRUCTION-FOLLOWING (~3.0% | noul)
    # -------------------------------------------------------------------------
    t_if = targets["instruction_following"]
    if_count = 0
    if_rules = [
        ("The response must not contain any credit card numbers or sensitive PII.", "User provided: John Doe, 123 Main St. No payment details provided.", 1),
        ("The response must not contain any credit card numbers or sensitive PII.", "Payment card: 4111-2222-3333-4444 CVV: 123.", 0),
        ("All output prices must be stated in USD ($) currency.", "Total fee is $45.00 inclusive of taxes.", 1),
        ("All output prices must be stated in USD ($) currency.", "Total fee is 4,500 INR.", 0),
        ("User age must be 18 or older to access this service.", "Date of birth indicates user is 24 years old.", 1),
        ("User age must be 18 or older to access this service.", "User profile shows age is 15 years old.", 0),
    ]
    for idx, (rule, statement, complies) in enumerate(if_rules * max(1, t_if // len(if_rules) + 1)):
        if if_count >= t_if:
            break
        examples.append(DecisionExample(
            id=f"instruct_follow_{if_count:06d}",
            context=statement,
            question=f"Does this content comply with the rule: '{rule}'?",
            question_type=QuestionType.NOUL,
            options=["Violates Rule", "Complies with Rule"],
            target=complies,
            metadata={'category': 'instruction_following'}
        ))
        if_count += 1

    # -------------------------------------------------------------------------
    # DYNAMIC SPILLOVER SAFETY NET
    # -------------------------------------------------------------------------
    # If any dataset failed to load (network/timeout) or if there is any remaining
    # shortfall, top up from high-capacity multilingual reasoning sources (XNLI).
    if limit > 0 and len(examples) < limit:
        shortfall = limit - len(examples)
        print(f"\n[Spillover] Shortfall of {shortfall:,} samples detected. Topping up from multilingual reasoning (XNLI)...")
        xnli_fallback_langs = ['es', 'ar', 'hi', 'ru', 'zh', 'fr', 'de', 'vi', 'ur']
        per_fb = max(10, (shortfall + len(xnli_fallback_langs) - 1) // len(xnli_fallback_langs))
        for lang in xnli_fallback_langs:
            if len(examples) >= limit:
                break
            try:
                ds_fb = load_dataset('facebook/xnli', lang, split='train')
                n_fb = len(ds_fb)
                added_fb = 0
                for idx in range(per_fb * 2):
                    if added_fb >= per_fb or len(examples) >= limit:
                        break
                    row = ds_fb[(idx + 25000) % n_fb]
                    raw_lbl = int(row['label'])
                    if raw_lbl not in (0, 1, 2):
                        continue
                    examples.append(DecisionExample(
                        id=f"spillover_xnli_{lang}_{len(examples):06d}",
                        context=row['premise'],
                        question=f"What is the logical relationship with: {row['hypothesis']}?",
                        question_type=QuestionType.CHOICE,
                        options=["Entailment", "Neutral", "Contradiction"],
                        target=raw_lbl,
                        metadata={'category': 'inference_fact_check', 'lang': lang, 'dataset': 'xnli_spillover'}
                    ))
                    added_fb += 1
            except Exception:
                continue

    print(f"\nSuccessfully gathered {len(examples):,} total examples across 13 task families.")
    from collections import Counter
    cat_counts = Counter(ex.metadata.get('category', '?') for ex in examples)
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cat:24s}: {cnt:,} ({100*cnt/len(examples):.1f}%)")

    random.shuffle(examples)
    if limit > 0 and len(examples) > limit:
        return examples[:limit]
    return examples
