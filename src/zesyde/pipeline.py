import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import onnxruntime as ort
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from zesyde.pipeline_utils import (
    DEFAULT_BACKBONE,
    DEFAULT_MULTILINGUAL_BACKBONE,
    EVAL_FILES,
    MULTILINGUAL_EVAL_FILES,
    make_collate,
    majority_baseline,
    collect_predictions,
    metrics_from_rows,
    derive_act_targets,
    train_act_head,
    fit_temperatures,
    write_sidecar,
    build_ort_session,
    rows_from_onnx,
    benchmark_latency,
    format_metrics,
    serialize_temperatures,
    onnx_forward_batch,
)
from zesyde.metrics import compute_metrics
from zesyde.rlcd_train import train_rlcd_epoch
from zesyde.collator import ZesydeCollator
from zesyde.data import load_frozen_dataset
from zesyde.onnx_export import export_to_onnx, quantize_to_int8
from zesyde.train import train_epoch
from zesyde.train_data import build_mixed_train
from zesyde.model import ZesydeModel
from zesyde.model_multilingual import ZesydeMultilingualModel

class BuildPipeline:
    def __init__(self, args):
        self.args = args
        if getattr(args, "backbone", None) is None:
            self.args.backbone = DEFAULT_MULTILINGUAL_BACKBONE if getattr(args, "multilingual", False) else DEFAULT_BACKBONE
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_sla = args.sla_latency or (96.0 if "large" in self.args.backbone else 46.0)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.collator = ZesydeCollator(tokenizer=self.tokenizer, max_length=args.max_length)
        
        self.calib_sets = {}
        self.test_sets = {}
        is_multi = getattr(args, "multilingual", False)
        if is_multi and Path("data/eval_multilingual").exists():
            self.eval_files = MULTILINGUAL_EVAL_FILES
        else:
            self.eval_files = EVAL_FILES
        self.train_examples = []
        self.model = None
        self.post_rows = {}
        self.calib_rows = []
        self.test_rows = []
        self.temperatures = {}
        self.fp32_session = None
        self.int8_session = None
        self.int8_latency = {}
        self.build_t0 = time.time()
        self.loader = None

        from zesyde.checkpoint import PauseController
        checkpoint_dir = getattr(args, "checkpoint_dir", None) or Path("models/checkpoints")
        self.pause_controller = PauseController(
            checkpoint_dir=checkpoint_dir,
            enable_interactive=getattr(args, "interactive_pause", True),
        )

    def init_model(self):
        if self.model is not None:
            return
        use_peft_flag = getattr(self.args, "use_peft", True)
        lora_r = getattr(self.args, "lora_r", 8)
        lora_alpha = getattr(self.args, "lora_alpha", 16)
        lora_dropout = getattr(self.args, "lora_dropout", 0.05)

        head_heads = getattr(self.args, "head_heads", None)
        if getattr(self.args, "multilingual", False):
            self.model = ZesydeMultilingualModel(
                backbone_name_or_path=self.args.backbone,
                head_num_heads=head_heads,
                use_peft=use_peft_flag,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.model = ZesydeModel(
                backbone_name_or_path=self.args.backbone,
                head_num_heads=head_heads,
                use_peft=use_peft_flag,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        self.model.to(self.device)


    def phase_0_data(self):
        for name, path in self.eval_files.items():
            examples = load_frozen_dataset(Path(path))[: self.args.eval_limit]
            self.calib_sets[name] = examples[0::2]
            self.test_sets[name] = examples[1::2]
        
        print("\n--- Held-out sets (calibration half / report half) ---")
        for name in self.eval_files:
            print(f"  {name:28s} calib={len(self.calib_sets[name]):4d}  report={len(self.test_sets[name]):4d}  majority={majority_baseline(self.test_sets[name]) * 100:.1f}%")

    def phase_1_warmstart(self, start_epoch: int = 1, start_step: int = 0, resume_optimizer_state=None, resume_scheduler_state=None):
        is_multi = getattr(self.args, "multilingual", False)
        if not self.train_examples:
            self.train_examples = build_mixed_train(limit=self.args.train_limit, multilingual=is_multi)
        print(f"\n--- Phase 1: supervised warm-start on {len(self.train_examples)} mixed examples ---")
        task_counts = Counter(ex.metadata.get("dataset", "?") for ex in self.train_examples)
        print("  task mix: " + ", ".join(f"{k}={v}" for k, v in sorted(task_counts.items())))

        self.init_model()
        if hasattr(self.model, "print_trainable_parameters") and start_epoch == 1 and start_step == 0:
            self.model.print_trainable_parameters()

        if start_epoch == 1 and start_step == 0:
            pre_rows = {name: collect_predictions(self.model, self.collator, ex, device=str(self.device)) for name, ex in self.test_sets.items()}
            print("\n  Zero-shot (random-init head) baseline:")
            for name, rows in pre_rows.items():
                print(f"    {name:28s} {format_metrics(metrics_from_rows(rows))}")

        if self.loader is None:
            self.loader = DataLoader(self.train_examples, batch_size=self.args.batch_size, shuffle=True, collate_fn=make_collate(self.collator))
            
        use_peft_flag = getattr(self.args, "use_peft", True)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        opt_groups = []
        if backbone_params:
            lr_backbone = getattr(self.args, "lr_lora", None) or getattr(self.args, "lr_backbone", 1e-4 if use_peft_flag else 1e-5)
            opt_groups.append({"params": backbone_params, "lr": lr_backbone})
        opt_groups.append({"params": self.model.option_head.parameters(), "lr": self.args.lr_head})
        opt_groups.append({"params": self.model.act_head.parameters(), "lr": self.args.lr_head})
        
        optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.1)
        total_steps = max(1, len(self.loader) * self.args.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
        if resume_scheduler_state is not None:
            scheduler.load_state_dict(resume_scheduler_state)

        for epoch in range(start_epoch, self.args.epochs + 1):
            step_offset = start_step if epoch == start_epoch else 0
            t0 = time.time()
            if train_epoch is None:
                raise NotImplementedError("train_epoch is not implemented or imported correctly.")
            avg_loss = train_epoch(
                self.model,
                self.loader,
                optimizer,
                scheduler=scheduler,
                device=str(self.device),
                smoothing=0.1,
                pause_controller=self.pause_controller,
                phase="phase_1_warmstart",
                epoch=epoch,
                total_epochs=self.args.epochs,
                start_step=step_offset,
                extra_state={"temperatures": self.temperatures},
            )
            print(f"  epoch {epoch}/{self.args.epochs} | loss={avg_loss:.4f} | {time.time() - t0:.0f}s")
            
            # Periodic checkpoint at end of epoch
            self.pause_controller.save_state(
                phase="phase_1_warmstart",
                epoch=epoch,
                total_epochs=self.args.epochs,
                step=len(self.loader),
                total_steps=len(self.loader),
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state={"temperatures": self.temperatures},
            )

    def phase_1_5_rlcd(self, start_epoch: int = 1, start_step: int = 0, resume_optimizer_state=None):
        if self.args.rlcd_epochs > 0:
            print(f"\n--- Phase 1.5: RLCD optimization with strictly proper scoring rules ({self.args.rlcd_epochs} epochs) ---")
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
                
            lr_rlcd = getattr(self.args, "lr_rlcd", None)
            if lr_rlcd is None:
                lr_rlcd = getattr(self.args, "lr_head", 3e-4) * 0.05
            rlcd_optimizer = torch.optim.AdamW(self.model.option_head.parameters(), lr=lr_rlcd, weight_decay=0.01)
            if resume_optimizer_state is not None:
                rlcd_optimizer.load_state_dict(resume_optimizer_state)

            if self.loader is None:
                is_multi = getattr(self.args, "multilingual", False)
                if not self.train_examples:
                    self.train_examples = build_mixed_train(limit=self.args.train_limit, multilingual=is_multi)
                self.loader = DataLoader(self.train_examples, batch_size=self.args.batch_size, shuffle=True, collate_fn=make_collate(self.collator))

            # Active validation hook evaluating raw accuracy and ECE periodically
            def active_val_hook(eval_model: ZesydeModel) -> Tuple[float, float]:
                eval_model.eval()
                all_preds, all_targets, all_probs = [], [], []
                with torch.no_grad():
                    for name, exs in self.calib_sets.items():
                        sub_exs = exs[:min(len(exs), 100)]
                        sub_loader = DataLoader(
                            sub_exs,
                            batch_size=min(32, len(sub_exs)),
                            shuffle=False,
                            collate_fn=make_collate(self.collator),
                        )
                        for batch in sub_loader:
                            out = eval_model(
                                input_ids=batch['input_ids'].to(self.device),
                                attention_mask=batch['attention_mask'].to(self.device),
                                marker_indices=batch['marker_indices'],
                            )
                            for i, logits in enumerate(out['logits']):
                                p = torch.softmax(logits, dim=-1).cpu().numpy()
                                all_preds.append(int(np.argmax(p)))
                                all_targets.append(int(batch['targets'][i].item()))
                                all_probs.append(p)
                eval_model.train()
                metrics = compute_metrics(all_preds, all_targets, all_probs)
                return float(metrics.get('accuracy', 0.0)), float(metrics.get('ece', 0.0))

            for r_ep in range(start_epoch, self.args.rlcd_epochs + 1):
                step_offset = start_step if r_ep == start_epoch else 0
                t0 = time.time()
                if train_rlcd_epoch is None:
                    raise NotImplementedError("train_rlcd_epoch is not implemented or imported correctly.")
                r_stats = train_rlcd_epoch(
                    self.model,
                    self.loader,
                    rlcd_optimizer,
                    ref_model=ref_model,
                    device=str(self.device),
                    pause_controller=self.pause_controller,
                    phase="phase_1_5_rlcd",
                    epoch=r_ep,
                    total_epochs=self.args.rlcd_epochs,
                    start_step=step_offset,
                    extra_state={"temperatures": self.temperatures},
                    kl_coef=getattr(self.args, "rlcd_kl_coef", 0.1),
                    max_grad_norm=getattr(self.args, "grad_clip", 1.0),
                    val_fn=active_val_hook,
                    eval_steps=getattr(self.args, "eval_steps", 100),
                    collapse_threshold=getattr(self.args, "collapse_threshold", 0.80),
                )
                print(f"  RLCD epoch {r_ep}/{self.args.rlcd_epochs} | loss={r_stats['loss']:.4f} | reward={r_stats['mean_reward']:.4f} | {time.time() - t0:.0f}s")
                self.pause_controller.save_state(
                    phase="phase_1_5_rlcd",
                    epoch=r_ep,
                    total_epochs=self.args.rlcd_epochs,
                    step=len(self.loader),
                    total_steps=len(self.loader),
                    model=self.model,
                    optimizer=rlcd_optimizer,
                    scheduler=None,
                    extra_state={"temperatures": self.temperatures},
                )
                if r_stats.get('early_stopped'):
                    print(f"  RLCD early stopped at epoch {r_ep} to preserve baseline accuracy.")
                    break


    def phase_2_evaluate(self):
        print("\n--- Phase 2: option head after training (report half, uncalibrated) ---")
        for name in self.eval_files:
            self.post_rows[name] = collect_predictions(self.model, self.collator, self.test_sets[name], device=str(self.device))
            self.calib_rows += collect_predictions(self.model, self.collator, self.calib_sets[name], device=str(self.device))
            self.test_rows += self.post_rows[name]
        for name, rows in self.post_rows.items():
            print(f"    {name:28s} {format_metrics(metrics_from_rows(rows))}")
        print(f"    {'OVERALL':28s} {format_metrics(metrics_from_rows(self.test_rows))}")

    def phase_3_calibrate(self):
        print("\n--- Phase 3: temperature calibration (fitted on calib half only) ---")
        self.temperatures = fit_temperatures(self.calib_rows)
        for key in sorted(self.temperatures, key=lambda k: (k[0], k[1])):
            print(f"    T[{key[0]}|{key[1]}] = {self.temperatures[key]:.4f}")

        print("\n  Calibration effect on report half:")
        for name in self.eval_files:
            before = metrics_from_rows(self.post_rows[name])
            after = metrics_from_rows(self.post_rows[name], self.temperatures)
            print(f"    {name:28s} ECE {before['ece']:.4f} -> {after['ece']:.4f} | Brier {before['brier_score']:.4f} -> {after['brier_score']:.4f}")
        
        overall_before = metrics_from_rows(self.test_rows)
        overall_calibrated = metrics_from_rows(self.test_rows, self.temperatures)
        print(f"    {'OVERALL':28s} ECE {overall_before['ece']:.4f} -> {overall_calibrated['ece']:.4f} | Brier {overall_before['brier_score']:.4f} -> {overall_calibrated['brier_score']:.4f}")

    def phase_4_act_head(self, start_epoch: int = 1):
        print("\n--- Phase 4: escalate head ---")
        if not self.train_examples:
            is_multi = getattr(self.args, "multilingual", False)
            self.train_examples = build_mixed_train(limit=self.args.train_limit, multilingual=is_multi)
        train_rows = collect_predictions(self.model, self.collator, self.train_examples, device=str(self.device))
        act_targets = derive_act_targets(train_rows, self.args.act_threshold)
        flag_rate = sum(act_targets) / max(1, len(act_targets))
        print(f"    frozen reference flags {flag_rate * 100:.1f}% of training rows for escalation (gold prob < {self.args.act_threshold} or wrong)")
        act_loss = train_act_head(
            self.model,
            self.collator,
            self.train_examples,
            act_targets,
            epochs=self.args.act_epochs,
            batch_size=self.args.batch_size,
            lr=self.args.lr_head,
            device=str(self.device),
            pause_controller=self.pause_controller,
            start_epoch=start_epoch,
            extra_state={"temperatures": self.temperatures},
        )
        print(f"    act head trained | final loss={act_loss:.4f}")

        self.pause_controller.save_state(
            phase="phase_4_act_head",
            epoch=self.args.act_epochs,
            total_epochs=self.args.act_epochs,
            step=1,
            total_steps=1,
            model=self.model,
            extra_state={"temperatures": self.temperatures},
        )

        self.test_rows = []
        self.post_rows = {}
        for name in self.eval_files:
            self.post_rows[name] = collect_predictions(self.model, self.collator, self.test_sets[name], device=str(self.device))
            self.test_rows += self.post_rows[name]
        
        ref_flags = derive_act_targets(self.test_rows, self.args.act_threshold)
        act_preds = [1 if r["act_logit"] > 0.0 else 0 for r in self.test_rows]
        tp = sum(1 for p, t in zip(act_preds, ref_flags) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(act_preds, ref_flags) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(act_preds, ref_flags) if p == 0 and t == 1)
        act_acc = sum(1 for p, t in zip(act_preds, ref_flags) if p == t) / max(1, len(ref_flags))
        print(f"    escalate vs frozen reference: acc={act_acc * 100:.1f}% precision={tp / max(1, tp + fp) * 100:.1f}% recall={tp / max(1, tp + fn) * 100:.1f}%")

    def phase_5_checkpoint(self):
        self.args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone_name": self.args.backbone,
            "model_state_dict": self.model.state_dict(),
            "temperatures": serialize_temperatures(self.temperatures),
            "config": {
                "max_length": self.args.max_length,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "lr_head": self.args.lr_head,
                "lr_backbone": self.args.lr_backbone,
                "act_threshold": self.args.act_threshold,
                "train_samples": len(self.train_examples),
                "use_peft": getattr(self.args, "use_peft", False),
                "lora_r": getattr(self.args, "lora_r", 8),
            },
        }, self.args.checkpoint)
        ckpt_mb = self.args.checkpoint.stat().st_size / (1024 * 1024)
        print(f"\n--- Phase 5: checkpoint saved -> {self.args.checkpoint} ({ckpt_mb:.1f} MB) ---")

    def phase_6_export(self):
        print("\n--- Phase 6: ONNX export + INT8 quantisation ---")
        if hasattr(self.model, "merge_peft"):
            self.model.merge_peft()
        self.model.eval()
        self.model.cpu()
        export_to_onnx(self.model, self.args.fp32_out, opset_version=18)
        print(f"    FP32: {self.args.fp32_out} ({self.args.fp32_out.stat().st_size / 1e6:.1f} MB)")
        quantize_to_int8(self.args.fp32_out, self.args.int8_out)
        if self.args.int8_out.exists():
            print(f"    INT8: {self.args.int8_out} ({self.args.int8_out.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"    INT8: Skipped or Failed")

    def phase_7_verify(self):
        print("\n--- Phase 7: verifying the served artifact ---")
        self.fp32_session = build_ort_session(self.args.fp32_out)
        self.serving_model_path = self.args.int8_out if self.args.int8_out.exists() else self.args.fp32_out
        self.serving_session = build_ort_session(self.serving_model_path)

        if self.serving_model_path == self.args.int8_out:
            print("  Verifying INT8 quantization parity...")
            probe_key = next(iter(self.test_sets))
            probe = self.test_sets[probe_key][:32]
            fp_scores, _ = onnx_forward_batch(self.fp32_session, self.collator, probe)
            i8_scores, _ = onnx_forward_batch(self.serving_session, self.collator, probe)
            agreement = sum(int(np.argmax(f) == np.argmax(i)) for f, i in zip(fp_scores, i8_scores)) / max(1, len(fp_scores))
            max_delta = max(float(np.max(np.abs(f - i))) for f, i in zip(fp_scores, i8_scores))
            print(f"    FP32 vs INT8 top-1 agreement: {agreement * 100:.1f}% | max |delta score| = {max_delta:.4f}")
        else:
            print("  Skipping INT8 parity check (using FP32 artifact for serving).")

        print("\n  Served accuracy on report half (temperature-calibrated):")
        served_rows = []
        accuracy_drifts = []
        for name in self.eval_files:
            rows = rows_from_onnx(self.serving_session, self.collator, self.test_sets[name])
            served_rows += rows
            torch_metrics = metrics_from_rows(self.post_rows[name], self.temperatures)
            serving_metrics = metrics_from_rows(rows, self.temperatures)
            accuracy_drifts.append(abs(torch_metrics["accuracy"] - serving_metrics["accuracy"]))
            print(f"    {name:28s} {format_metrics(serving_metrics)}  (torch {torch_metrics['accuracy'] * 100:.1f}% -> served {serving_metrics['accuracy'] * 100:.1f}%)")
        
        served_metrics = metrics_from_rows(served_rows, self.temperatures)
        print(f"    {'OVERALL':28s} {format_metrics(served_metrics)}")
        if accuracy_drifts:
            print(f"    max per-task accuracy drift from quantisation/export: {max(accuracy_drifts) * 100:.1f} pp")

    def phase_8_latency(self):
        first_key = next(iter(self.eval_files))
        latency_batch = self.collator([self.test_sets[first_key][0]])
        latency_inputs = {
            "input_ids": latency_batch["input_ids"].numpy(),
            "attention_mask": latency_batch["attention_mask"].numpy(),
            "marker_positions": np.array([latency_batch["marker_indices"][0]], dtype=np.int64),
        }
        fp32_latency = benchmark_latency(self.fp32_session, latency_inputs, self.args.benchmark_runs)
        self.serving_latency = benchmark_latency(self.serving_session, latency_inputs, self.args.benchmark_runs)
        print("\n--- Phase 8: batch-1 CPU latency ---")
        print(f"    FP32: p50={fp32_latency['p50']:.2f} ms | p99={fp32_latency['p99']:.2f} ms")
        if self.serving_model_path == self.args.int8_out:
            print(f"    INT8: p50={self.serving_latency['p50']:.2f} ms | p99={self.serving_latency['p99']:.2f} ms | mean={self.serving_latency['mean']:.2f} ms")

        write_sidecar(self.args.calibration, self.temperatures, backbone_name=self.args.backbone)
        print(f"\nCalibration sidecar -> {self.args.calibration}")
        print(f"Total build time: {time.time() - self.build_t0:.0f}s")

    def print_verdict(self):
        print("\n--- Build verdict ---")
        for name, rows in self.post_rows.items():
            baseline = majority_baseline(self.test_sets[name])
            served_acc = metrics_from_rows(rows, self.temperatures)["accuracy"]
            flag = "BEATS" if served_acc > baseline else "below"
            print(f"  {name:28s} {flag} majority baseline ({served_acc * 100:.1f}% vs {baseline * 100:.1f}%)")
        
        if self.serving_latency["p99"] >= self.target_sla:
            print(f"  [Notice] p99 latency ({self.serving_latency['p99']:.2f}ms) exceeds target SLA ({self.target_sla:.2f}ms)")
        else:
            print(f"  [Pass] p99 latency ({self.serving_latency['p99']:.2f}ms) satisfies target SLA ({self.target_sla:.2f}ms)")

    def resume_from_checkpoint(self, checkpoint_path=None):
        from zesyde.checkpoint import PauseController
        target_path = checkpoint_path or getattr(self.args, "resume_checkpoint", None) or self.pause_controller.checkpoint_path
        target_path = Path(target_path)
        print(f"\n🔄 Resuming training from checkpoint: {target_path}")
        ckpt = PauseController.load_state(target_path, device=self.device)
        PauseController.restore_rng(ckpt.get("rng", {}))

        self.phase_0_data()
        self.init_model()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.temperatures = ckpt.get("extra_state", {}).get("temperatures", {})

        phase = ckpt.get("phase", "phase_1_warmstart")
        saved_epoch = ckpt.get("epoch", 1)
        saved_step = ckpt.get("step", 0)
        total_steps = ckpt.get("total_steps", 0)

        print(f"    Loaded state: Phase '{phase}', Epoch {saved_epoch}, Step {saved_step}/{total_steps}")

        if phase == "phase_1_warmstart":
            is_epoch_done = (saved_step >= total_steps) if total_steps > 0 else False
            start_epoch = saved_epoch + 1 if is_epoch_done else saved_epoch
            start_step = 0 if is_epoch_done else saved_step

            if start_epoch <= self.args.epochs:
                self.phase_1_warmstart(
                    start_epoch=start_epoch,
                    start_step=start_step,
                    resume_optimizer_state=ckpt.get("optimizer_state_dict") if not is_epoch_done else None,
                    resume_scheduler_state=ckpt.get("scheduler_state_dict") if not is_epoch_done else None,
                )
            else:
                print("    Phase 1 warmstart already completed in checkpoint. Proceeding to Phase 1.5...")

            self.phase_1_5_rlcd()
            self.phase_2_evaluate()
            self.phase_3_calibrate()
            self.phase_4_act_head()
            self.phase_5_checkpoint()
            self.phase_6_export()
            self.phase_7_verify()
            self.phase_8_latency()
            self.print_verdict()

        elif phase == "phase_1_5_rlcd":
            is_epoch_done = (saved_step >= total_steps) if total_steps > 0 else False
            start_epoch = saved_epoch + 1 if is_epoch_done else saved_epoch
            start_step = 0 if is_epoch_done else saved_step

            if start_epoch <= self.args.rlcd_epochs:
                self.phase_1_5_rlcd(
                    start_epoch=start_epoch,
                    start_step=start_step,
                    resume_optimizer_state=ckpt.get("optimizer_state_dict") if not is_epoch_done else None,
                )
            else:
                print("    Phase 1.5 RLCD already completed in checkpoint. Proceeding to Phase 2...")

            self.phase_2_evaluate()
            self.phase_3_calibrate()
            self.phase_4_act_head()
            self.phase_5_checkpoint()
            self.phase_6_export()
            self.phase_7_verify()
            self.phase_8_latency()
            self.print_verdict()

        elif phase == "phase_4_act_head":
            is_epoch_done = (saved_step >= total_steps) if total_steps > 0 else False
            start_epoch = saved_epoch + 1 if is_epoch_done else saved_epoch

            if start_epoch <= self.args.act_epochs:
                self.phase_4_act_head(start_epoch=start_epoch)
            else:
                print("    Phase 4 Act head already completed in checkpoint. Proceeding to Phase 5...")

            self.phase_5_checkpoint()
            self.phase_6_export()
            self.phase_7_verify()
            self.phase_8_latency()
            self.print_verdict()

        else:
            print(f"Unknown phase '{phase}'. Resuming from Phase 2...")
            self.phase_2_evaluate()
            self.phase_3_calibrate()
            self.phase_4_act_head()
            self.phase_5_checkpoint()
            self.phase_6_export()
            self.phase_7_verify()
            self.phase_8_latency()
            self.print_verdict()

    def run_all(self):
        from zesyde.checkpoint import TrainingInterruptedException
        try:
            if getattr(self.args, "resume", False) or getattr(self.args, "resume_checkpoint", None) is not None:
                self.resume_from_checkpoint(getattr(self.args, "resume_checkpoint", None))
                return

            print("=== Zesyde Production Model Build ===")
            print(f"Backbone: {self.args.backbone} | Device: {self.device} | max_length={self.args.max_length} | SLA={self.target_sla:.1f}ms")
            self.phase_0_data()
            self.phase_1_warmstart()
            self.phase_1_5_rlcd()
            self.phase_2_evaluate()
            self.phase_3_calibrate()
            self.phase_4_act_head()
            self.phase_5_checkpoint()
            self.phase_6_export()
            self.phase_7_verify()
            self.phase_8_latency()
            self.print_verdict()
        except TrainingInterruptedException as e:
            print(f"\n{e}")
            print("👉 To resume training anytime, run:")
            print("   python scripts/build_prod_model.py --resume\n")

