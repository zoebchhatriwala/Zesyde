"""Tokenizer and sequence collator for Zesyde option-marker formatting."""
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zesyde.schema import DecisionExample

class ZesydeCollator:
    """Formats DecisionExamples into tokenized sequences with option marker tracking.

    Sequence structure:
    [CLS] context [SEP] question [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP]
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int = 8192, marker_token: str = "[MASK]"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_token = marker_token
        self.marker_token_id = tokenizer.convert_tokens_to_ids(marker_token)
        if self.marker_token_id is None or self.marker_token_id == tokenizer.unk_token_id:
            self.marker_token_id = tokenizer.mask_token_id

    def __call__(self, examples: List[DecisionExample]) -> Dict[str, Any]:
        batch_input_ids = []
        batch_marker_indices = []
        targets = []
        
        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for ex in examples:
            ctx_text = ex.context.strip() if ex.context else ""
            q_text = ex.question.strip() if ex.question else ""

            ctx_ids = self.tokenizer.encode(ctx_text, add_special_tokens=False) if ctx_text else []
            q_ids = self.tokenizer.encode(q_text, add_special_tokens=False) if q_text else []
            
            opt_segment_ids = []
            opt_marker_rel_positions = []
            
            for opt in ex.options:
                marker_pos = len(opt_segment_ids)
                opt_marker_rel_positions.append(marker_pos)
                opt_segment_ids.append(self.marker_token_id)
                opt_tokens = self.tokenizer.encode(opt, add_special_tokens=False)
                opt_segment_ids.extend(opt_tokens)
                
            if ctx_ids and q_ids:
                prefix_ids = [cls_id] + ctx_ids + [sep_id] + q_ids + [sep_id]
            elif ctx_ids:
                prefix_ids = [cls_id] + ctx_ids + [sep_id]
            elif q_ids:
                prefix_ids = [cls_id] + q_ids + [sep_id]
            else:
                prefix_ids = [cls_id, sep_id]

            total_len = len(prefix_ids) + len(opt_segment_ids) + 1
            
            if total_len > self.max_length:
                overflow = total_len - self.max_length
                if len(ctx_ids) > overflow:
                    ctx_ids = ctx_ids[:-overflow]
                    if q_ids:
                        prefix_ids = [cls_id] + ctx_ids + [sep_id] + q_ids + [sep_id]
                    else:
                        prefix_ids = [cls_id] + ctx_ids + [sep_id]
                else:
                    prefix_ids = prefix_ids[:max(1, self.max_length - len(opt_segment_ids) - 1)]
            
            prefix_len = len(prefix_ids)
            abs_markers = [pos + prefix_len for pos in opt_marker_rel_positions]
            
            full_seq = prefix_ids + opt_segment_ids + [sep_id]
            batch_input_ids.append(full_seq)
            batch_marker_indices.append(abs_markers)
            targets.append(ex.target)
            
        max_batch_len = max(len(seq) for seq in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for seq in batch_input_ids:
            pad_len = max_batch_len - len(seq)
            padded_input_ids.append(seq + [pad_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)
            
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'marker_indices': batch_marker_indices,
            'targets': torch.tensor(targets, dtype=torch.long),
            'teacher_probs': [torch.tensor(ex.teacher_probs, dtype=torch.float32) if ex.teacher_probs is not None else None for ex in examples]
        }

