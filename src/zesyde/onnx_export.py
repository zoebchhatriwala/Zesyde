"""ONNX Export logic for Zesyde."""
import torch
from pathlib import Path
from zesyde.model import ZesydeModel

from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(model: ZesydeModel, output_path: Path, opset_version: int = 18) -> None:
    """Exports a trained ZesydeModel to ONNX format."""
    if hasattr(model, "merge_peft"):
        model.merge_peft()
    model.eval()
    
    # We create a wrapper class to handle the tensor extraction for ONNX
    class ONNXZesydeWrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core_model = core_model
            
        def forward(self, input_ids, attention_mask, marker_positions):
            outputs = self.core_model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            
            # Vectorized mean-pooling of option tokens
            device = last_hidden.device
            B = last_hidden.size(0)
            S = last_hidden.size(1)
            _, M = marker_positions.shape
            
            seq_grid = torch.arange(S, device=device).view(1, 1, S)
            start_mask = seq_grid >= marker_positions.unsqueeze(-1)
            
            seq_lens = attention_mask.sum(dim=-1, keepdim=True)
            next_positions = torch.cat([marker_positions[:, 1:], seq_lens - 1], dim=1)
            next_markers = torch.where(next_positions <= marker_positions, seq_lens, next_positions)
            end_mask = seq_grid < next_markers.unsqueeze(-1)
            
            option_mask = (start_mask & end_mask).float() * attention_mask.unsqueeze(1).float()
            mask_sums = option_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            
            opt_hidden = torch.bmm(option_mask, last_hidden) / mask_sums
            cls_hidden = last_hidden[:, 0, :]
            
            if getattr(self.core_model, 'use_context_fusion', True):
                cls_expanded = cls_hidden.unsqueeze(1).expand(-1, M, -1)
                features = torch.cat([opt_hidden, cls_expanded, opt_hidden * cls_expanded], dim=-1)
            else:
                features = opt_hidden
            
            opt_scores = self.core_model.option_head(features).squeeze(-1)
            act_logits = self.core_model.act_head(cls_hidden).squeeze(-1)
            
            return opt_scores, act_logits

    wrapper = ONNXZesydeWrapper(model)
    wrapper.eval()
    dummy_input_ids = torch.zeros((1, 128), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)
    dummy_marker_positions = torch.tensor([[10, 20]], dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_marker_positions),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'marker_positions'],
        output_names=['logits', 'act_logits'],
        dynamo=False,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'marker_positions': {0: 'batch_size', 1: 'num_options'},
            'logits': {0: 'batch_size', 1: 'num_options'},
            'act_logits': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {output_path}")

def quantize_to_int8(input_path: Path, output_path: Path) -> None:
    """Quantizes an ONNX model to dynamic INT8, preserving decision heads in FP32."""
    try:
        import onnx
        model = onnx.load(str(input_path))
        head_nodes = [n.name for n in model.graph.node if 'option_head' in n.name or 'act_head' in n.name]
        
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=head_nodes if head_nodes else None,
        )
        print(f"Quantized ONNX model to {output_path} (preserved {len(head_nodes)} head nodes in FP32)")
    except Exception as e:
        print(f"Warning: INT8 quantization failed ({e}). Proceeding without quantized model.")

# Alias for backwards compatibility with tests and callers
export_model = export_to_onnx
