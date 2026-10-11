"""Fast sub-millisecond Language Router for Zesyde.

Laya-style routing architecture:
- Sub-millisecond Unicode script & language detection (standard library, zero extra dependencies)
- Dynamic dispatch to language-specialized or multilingual ZesydeRouter instances
- Robust fallback to multilingual checkpoint for mixed-script or unseen languages
"""
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from zesyde.router import ZesydeRouter, DecisionResult
from zesyde.schema import QuestionType

_EN_STOPWORDS = frozenset({'we', 'can', 'look', 'not', 'now', 'you', 'on', 'do', 'her', 'that', 'their', 'of', 'or', 'will', 'i', 'make', 'with', 'go', 'into', 'get', 'two', 'his', 'how', 'as', 'but', 'for', 'just', 'which', 'because', 'some', 'than', 'to', 'and', 'when', 'see', 'by', 'my', 'only', 'new', 'me', 'them', 'also', 'there', 'the', 'think', 'so', 'at', 'even', 'a', 'no', 'he', 'work', 'be', 'your', 'then', 'know', 'out', 'an', 'first', 'well', 'year', 'one', 'if', 'they', 'time', 'use', 'all', 'this', 'say', 'want', 'back', 'its', 'come', 'way', 'have', 'she', 'from', 'who', 'could', 'good', 'it', 'about', 'take', 'over', 'people', 'him', 'after', 'up', 'like', 'our', 'what', 'in', 'other', 'would'})
_FR_STOPWORDS = frozenset({'par', 'des', 'que', 'et', 'pour', 'cette', 'aux', 'ne', 'plus', 'dans', 'qui', 'une', 'sur', 'se', 'mais', 'les', 'du', 'avec', 'pas', 'ou', 'en', 'la', 'ce', 'nous', 'au', 'un', 'le', 'de', 'sont', 'est'})
_DE_STOPWORDS = frozenset({'mit', 'des', 'einer', 'als', 'und', 'auf', 'an', 'wie', 'eines', 'nicht', 'für', 'den', 'im', 'ein', 'zu', 'einen', 'dem', 'einem', 'der', 'es', 'nach', 'in', 'eine', 'die', 'ist', 'das', 'von', 'auch'})
_ES_STOPWORDS = frozenset({'y', 'su', 'que', 'del', 'se', 'para', 'los', 'e', 'les', 'el', 'por', 'en', 'como', 'lo', 'una', 'la', 'al', 'unas', 'es', 'un', 'le', 'mas', 'de', 'unos', 'las', 'no', 'con', 'sus', 'pero'})

def detect_script(text: str) -> str:
    if not text:
        return 'LATIN'
    script_counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
            script = name.split()[0]
            if 'CJK' in name or 'IDEOGRAPH' in name or 'HANGUL' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                script = 'CJK'
            elif 'ARABIC' in name:
                script = 'ARABIC'
            elif 'DEVANAGARI' in name:
                script = 'DEVANAGARI'
            elif 'CYRILLIC' in name:
                script = 'CYRILLIC'
            elif 'GREEK' in name:
                script = 'GREEK'
            elif 'THAI' in name:
                script = 'THAI'
            elif 'LATIN' in name:
                script = 'LATIN'
                
            script_counts[script] = script_counts.get(script, 0) + 1
        except ValueError:
            pass
            
    if not script_counts:
        return 'LATIN'
    return max(script_counts.items(), key=lambda x: x[1])[0]

def detect_language(text: str) -> str:
    script = detect_script(text)
    if script == 'CJK':
        return 'zh'
    if script == 'ARABIC':
        return 'ar'
    if script == 'DEVANAGARI':
        return 'hi'
    if script == 'CYRILLIC':
        return 'ru'
    if script == 'GREEK':
        return 'el'
    if script == 'THAI':
        return 'th'
    if script == 'LATIN':
        tokens = [w.strip('.,!?;:"\'()[]{}').lower() for w in text.split() if w.strip('.,!?;:"\'()[]{}').lower()]
        if not tokens:
            return 'en'
        scores = {
            'en': sum(1 for t in tokens if t in _EN_STOPWORDS),
            'fr': sum(1 for t in tokens if t in _FR_STOPWORDS),
            'de': sum(1 for t in tokens if t in _DE_STOPWORDS),
            'es': sum(1 for t in tokens if t in _ES_STOPWORDS)
        }
        best_lang, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score > 0:
            return best_lang
        return 'en'
    return 'en'

@dataclass
class MultilingualDecisionResult(DecisionResult):
    """Zesyde DecisionResult enriched with language routing metadata."""
    detected_language: str = 'en'
    detected_script: str = 'LATIN'
    routed_model: str = 'default'
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['detected_language'] = self.detected_language
        d['detected_script'] = self.detected_script
        d['routed_model'] = self.routed_model
        return d

class LanguageRouter:
    """Dispatches inputs to specialized language checkpoints or multilingual backbone.
    
    Architecture:
        1. Fast Unicode/heuristic language detection (< 0.1ms).
        2. Routing table lookup (maps lang code / script to target ZesydeRouter).
        3. Calibrated decision execution via selected model.
        4. Returns enriched MultilingualDecisionResult.
    """
    def __init__(self, default_router_key: str = 'multilingual', routers: Dict[str, ZesydeRouter] = None):
        if not routers:
            raise ValueError('Must provide at least one ZesydeRouter instance.')
        self.routers = routers
        if default_router_key not in routers:
            self.default_router_key = list(routers.keys())[0]
        else:
            self.default_router_key = default_router_key

    @classmethod
    def from_checkpoints(
        cls,
        en_path: Optional[Union[str, Path]] = None,
        multilingual_path: Optional[Union[str, Path]] = None,
        custom_routers: Optional[Dict[str, Union[str, Path]]] = None,
        tokenizer_name: str = 'jhu-clsp/mmBERT-base',
        max_length: int = 8192,
        prefer_fp32: bool = True,
        default_key: str = 'multilingual'
    ) -> 'LanguageRouter':
        routers = {}
        if en_path:
            try:
                routers['en'] = ZesydeRouter.from_pretrained(
                    en_path, tokenizer_name=tokenizer_name, max_length=max_length, prefer_fp32=prefer_fp32
                )
            except Exception:
                pass
        if multilingual_path:
            try:
                routers['multilingual'] = ZesydeRouter.from_pretrained(
                    multilingual_path, tokenizer_name=tokenizer_name, max_length=max_length, prefer_fp32=prefer_fp32
                )
            except Exception:
                pass
        if custom_routers:
            for key, path in custom_routers.items():
                try:
                    routers[key] = ZesydeRouter.from_pretrained(
                        path, tokenizer_name=tokenizer_name, max_length=max_length, prefer_fp32=prefer_fp32
                    )
                except Exception:
                    pass
        if not routers:
            raise FileNotFoundError('No valid model checkpoints found to initialize LanguageRouter.')
        return cls(default_router_key=default_key, routers=routers)

    def route_for_text(self, text: str) -> Tuple[ZesydeRouter, str, str, str]:
        script = detect_script(text)
        lang = detect_language(text)
        if lang in self.routers:
            return self.routers[lang], lang, lang, script
        
        if script.lower() in self.routers:
            return self.routers[script.lower()], script.lower(), lang, script
            
        default_r = self.routers[self.default_router_key]
        return default_r, self.default_router_key, lang, script

    def decide(
        self,
        context: str,
        options: List[str],
        question: str = 'Select the best option:',
        question_type: Union[QuestionType, str] = 'choice',
        escalation_threshold: float = 0.10,
        force_language: Optional[str] = None
    ) -> MultilingualDecisionResult:
        t0 = time.perf_counter()
        
        if context and any(c.isalpha() for c in context):
            sample_text = context
        else:
            sample_text = question
            
        if force_language and force_language in self.routers:
            router = self.routers[force_language]
            routed_key = force_language
            detected_lang = force_language
            detected_sc = detect_script(sample_text)
        else:
            router, routed_key, detected_lang, detected_sc = self.route_for_text(sample_text)
            
        res = router.decide(
            context=context,
            options=options,
            question=question,
            question_type=question_type,
            escalation_threshold=escalation_threshold
        )
        
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return MultilingualDecisionResult(
            selected_option=res.selected_option,
            confidence=res.confidence,
            probabilities=res.probabilities,
            should_escalate=res.should_escalate,
            escalate_probability=res.escalate_probability,
            latency_ms=round(total_latency_ms, 2),
            question_type=res.question_type,
            options=res.options,
            detected_language=detected_lang,
            detected_script=detected_sc,
            routed_model=routed_key
        )
