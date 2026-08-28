# Import necessary libraries
import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

import torch
import gc
import torch.nn as nn
import math
import json
import os
import re

from typing import Dict, Optional, Tuple, List
from torch.optim import Optimizer

import nltk
nltk.download('punkt', quiet=True)

from peft.tuners.lora import LoraLayer
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
)

# === 1. The K-FAC Data Collector ===
class KFACAutogradFunction(torch.autograd.Function):
    """Custom autograd function to capture activations and gradients for K-FAC."""

    @staticmethod
    def forward(ctx, module, output, input):
        ctx.module = module
        ctx.save_for_backward(input.detach())
        return output

    @staticmethod
    def backward(ctx, grad_wrt_output):
        module = ctx.module
        input, = ctx.saved_tensors
        manager = module.grit_manager

        if not module.training:
            return None, grad_wrt_output, None

        # SPEED + ABLATION: skip covariance accumulation only when NEITHER mechanism that
        # reads it is active AND shadow-tracking-for-analysis isn't requested — both
        # precondition_gradients() (natural-gradient step) and neural_reprojection() (rank
        # pruning) depend on a_cov/g_cov, so tracking must stay on if either is enabled, or
        # if track_covariance_for_analysis=True is deliberately measuring curvature on a
        # LoRA-only run for spectral comparison against a real GRIT run.
        if (getattr(manager.config, "disable_precondition", False)
                and getattr(manager.config, "disable_reprojection", False)
                and not getattr(manager.config, "track_covariance_for_analysis", False)):
            return None, grad_wrt_output, None

        tick = manager.cov_update_tick.get(module, 0) + 1
        manager.cov_update_tick[module] = tick

        if tick % manager.config.grit_cov_update_freq != 0:
            return None, grad_wrt_output, None

        with torch.no_grad():
            rho = float(getattr(manager.config, 'kfac_ema_momentum', 0.95))

            # --- A-Cov ---
            a = input.reshape(-1, input.shape[-1])
            if a.shape[0] > 0 and 'default' in module.lora_A:
                grad_device = grad_wrt_output.device
                lora_A_T = module.lora_A['default'].weight.data.T.to(
                    device=grad_device, dtype=a.dtype, non_blocking=True)
                projected_a = a @ lora_A_T
                # FIX (confirmed via smoke test): was a raw, unnormalized sum of outer
                # products over projected_a.shape[0] rows — magnitude scaled with however
                # many activation rows were in this micro-batch (up to ~max_length),
                # swamping kfac_damping and dominating compute_loss()'s curvature
                # regularizer by up to ~30x the real CE loss. Dividing by row count turns
                # this into an actual per-token covariance estimate.
                a_cov_sample = (projected_a.T @ projected_a) / max(1, projected_a.shape[0])

                current_cov = manager.a_covs[module].to(grad_device)
                n = manager.num_samples_a[module]
                manager.num_samples_a[module] = n + projected_a.shape[0]
                if n == 0:
                    updated_cov = a_cov_sample.float()
                else:
                    updated_cov = rho * current_cov.float() + (1.0 - rho) * a_cov_sample.float()
                # BUG 1 FIX: dtype is float32 — cast is a confirmed no-op
                manager.a_covs[module].copy_(updated_cov.to(dtype=manager.a_covs[module].dtype))

            # --- G-Cov ---
            g = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
            if g.shape[0] > 0 and 'default' in module.lora_B:
                lora_B = module.lora_B['default'].weight.data.to(
                    device=g.device, dtype=g.dtype, non_blocking=True)
                projected_g = g @ lora_B
                # Same normalization fix as a_cov_sample above.
                g_cov_sample = (projected_g.T @ projected_g) / max(1, projected_g.shape[0])

                current_cov = manager.g_covs[module].to(g_cov_sample.device)
                n = manager.num_samples_g[module]
                manager.num_samples_g[module] = n + projected_g.shape[0]
                if n == 0:
                    updated_cov = g_cov_sample.float()
                else:
                    updated_cov = rho * current_cov.float() + (1.0 - rho) * g_cov_sample.float()
                # BUG 1 FIX: dtype is float32 — cast is a confirmed no-op
                manager.g_covs[module].copy_(updated_cov.to(dtype=manager.g_covs[module].dtype))

        return None, grad_wrt_output, None


# === 2. The GRIT Configuration ===
class GRITConfig:
    """Configuration for GRIT training on Llama models."""

    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.batch_size = 1
        self.gradient_accumulation_steps = 8
        self.num_epochs = 3
        self.learning_rate = 1e-4
        self.precision = "bf16"
        self.max_length = 512

        # --- LoRA ---
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2"
        ]
        self.apply_lora_to_vision = False
        self.vision_lora_modules = []

        # --- K-FAC ---
        self.kfac_update_freq = 50
        self.kfac_damping = 0.005
        self.kfac_damping_ceiling = 1.0   # see step()'s adaptive-damping fix note

        # --- Ablation switches (for isolating which GRIT mechanism drives any observed
        # effect, rather than only being able to test "GRIT fully on" vs "fully off") ---
        self.disable_precondition = False           # natural-gradient step on A/B grads
        self.disable_reprojection = False            # periodic low-rank projection of A/B
        self.disable_curvature_regularizer = False   # lambda_kfac * curv_reg loss term
        self.disable_reprojection_regularizer = False  # lambda_reproj * reproj_reg loss term
        # When True, K-FAC covariance is still tracked (and telemetry.json still gets real
        # eigenvalues/effective ranks) even if disable_precondition and disable_reprojection
        # are both True — i.e. a LoRA-only run that passively measures what GRIT's curvature
        # statistics WOULD have looked like, without ever acting on them. Lets you compare
        # a vanilla-LoRA solution's implied curvature structure against a real GRIT run's,
        # using the same plot_heatmaps.py/plot_spectra.py/plot_spectra_compare.py tooling.
        self.track_covariance_for_analysis = False
        self.kfac_min_samples = 256
        self.grit_cov_update_freq = 8   # Q4 FIX: less aggressive for 11B
        self.kfac_ema_momentum = 0.95

        # --- Reprojection ---
        self.reprojection_freq = 200
        self.reprojection_k = 8
        self.use_two_sided_reprojection = True

        # --- Rank adaptation ---
        self.enable_rank_adaptation = True
        self.rank_adaptation_threshold = 0.90
        self.min_lora_rank = 4

        # --- Warmup ---
        self.regularizer_warmup_steps = 100
        self.reprojection_warmup_steps = 200
        self.rank_adaptation_start_step = 200
        self.ng_warmup_steps = 50

        # --- Dataset ---
        self.dataset_name = "ChongyanChen/VQAonline"
        self.num_workers = 2
        self.pin_memory = True
        self.drop_last = True

        # --- Loss lambdas ---
        self.lambda_kfac = 5e-6
        self.lambda_reproj = 5e-5

        # --- Logging ---
        self.log_fisher_spectrum = True
        self.log_top_eigs = 8
        self.kfac_inversion_device = 'cpu'
        self.log_final_eff_rank = True

        # --- VQA ---
        self.answer_max_length = 128
        self.ignore_index = -100

        # --- Optimizer ---
        self.use_8bit_adam = True
        self.gradient_checkpointing = True

        # --- Eval / generation ---
        self.gen_max_new_tokens = 128
        self.gen_do_sample = False

        # --- Telemetry (for plot_heatmaps.py and plot_spectra.py) ---
        self.telemetry_dir = "./results_vqa_grit_llama3_2_vision"


# === 3. Telemetry export ===
def _parse_lora_name(lora_name: str):
    """
    Extract layer index and module type from a LoRA module's full dotted name.
    e.g. 'base_model.model.model.language_model.layers.5.self_attn.q_proj'
         -> layer=5, module_type='q_proj'
    Returns (None, None) when the pattern is absent.
    """
    layer_match = re.search(r'\.layers\.(\d+)\.', lora_name)
    layer = int(layer_match.group(1)) if layer_match else None
    parts = lora_name.rstrip('.').split('.')
    module_type = parts[-1] if parts else None
    return layer, module_type


def _export_telemetry(manager, run_dir: str):
    """
    Write telemetry.json consumed by plot_heatmaps.py and plot_spectra.py.
    Called automatically at end of training from log_final_raw_ranks().

    Schema:
    {
      "global_step": int,
      "modules": [
        {
          "lora_name":   str,
          "layer":       int | null,
          "module_type": str | null,
          "k_a":         int,          # final A-side adapted rank
          "k_b":         int,          # final B-side adapted rank
          "eff_rank_a":  float,        # entropy-based effective rank of A-cov
          "eff_rank_g":  float,
          "eigvals_a":   [float, ...], # top-N eigenvalues A-side
          "eigvals_g":   [float, ...], # top-N eigenvalues G-side
        }, ...
      ]
    }
    """
    os.makedirs(run_dir, exist_ok=True)
    telemetry_path = os.path.join(run_dir, "telemetry.json")
    top_k = int(getattr(manager.config, 'log_top_eigs', 8))

    modules_data = []
    for module in manager.optimized_modules:
        name = getattr(module, 'lora_name', '<unknown>')
        layer, module_type = _parse_lora_name(name)
        base_r = int(module.r['default']) if hasattr(module, 'r') and 'default' in module.r else 0

        Va = manager._last_Va_k.get(module, None)
        Vg = manager._last_Vg_k.get(module, None)
        k_a = int(Va.shape[1]) if Va is not None and Va.numel() > 0 else base_r
        k_b = int(Vg.shape[1]) if Vg is not None and Vg.numel() > 0 else k_a

        eff_rank_a = manager._compute_effective_rank(
            manager.a_covs[module].detach().cpu().float())
        eff_rank_g = manager._compute_effective_rank(
            manager.g_covs[module].detach().cpu().float())

        eigvals_a, eigvals_g = [], []
        try:
            ev = torch.linalg.eigvalsh(manager.a_covs[module].detach().cpu().float())
            eigvals_a = [round(float(x), 6)
                         for x in ev.sort(descending=True).values[:top_k].tolist()]
        except Exception:
            pass
        try:
            ev = torch.linalg.eigvalsh(manager.g_covs[module].detach().cpu().float())
            eigvals_g = [round(float(x), 6)
                         for x in ev.sort(descending=True).values[:top_k].tolist()]
        except Exception:
            pass

        modules_data.append({
            "lora_name":   name,
            "layer":       layer,
            "module_type": module_type,
            "k_a":         k_a,
            "k_b":         k_b,
            "eff_rank_a":  round(eff_rank_a, 4),
            "eff_rank_g":  round(eff_rank_g, 4),
            "eigvals_a":   eigvals_a,
            "eigvals_g":   eigvals_g,
        })

    payload = {"global_step": manager.global_step, "modules": modules_data}
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[GRIT] Telemetry written → {telemetry_path}")
    print(f"[GRIT] Visualise with:")
    print(f"       python plot_heatmaps.py --run_dir {run_dir}")
    print(f"       python plot_spectra.py  --run_dir {run_dir}")


# === 4. The GRIT Manager ===
class GRITManager:
    """Manager for GRIT on Llama models."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.global_step = 0
        self.backward_step = 0

        self.loss_history = []
        self.loss_history_capacity = 20
        self.last_kfac_update_step = 0
        self.last_reprojection_step = 0

        # BUG 2 FIX: capture initial configured damping as floor
        self._initial_kfac_damping = float(config.kfac_damping)

        self.a_covs = {}
        self.g_covs = {}
        self.a_invs = {}
        self.g_invs = {}
        self.num_samples_a = {}
        self.num_samples_g = {}
        self.cov_update_tick = {}

        self.optimized_modules = []
        self.factors_are_ready = False

        self._last_Va_k = {}
        self._last_Vg_k = {}

        self._instrument_model()
        print("GRITManager: Initialization complete for Llama-3.2-11B model.")
        print(f"Optimizing {len(self.optimized_modules)} LoRA modules across vision + language towers.")

    def _instrument_model(self):
        print("Instrumenting model for GRIT...")
        vision_modules = 0
        language_modules = 0

        for name, module in self.model.named_modules():
            if (isinstance(module, LoraLayer)
                    and module.r.get('default', 0) > 0
                    and isinstance(module.base_layer, nn.Linear)):
                module.lora_name = name
                module.grit_manager = self

                is_vision = name.startswith('model.vision_model')

                self.num_samples_a[module] = 0
                self.num_samples_g[module] = 0
                self.cov_update_tick[module] = 0

                module.original_forward = module.forward

                def new_forward(self, x=None, *args, **kwargs):
                    if x is None:
                        x = kwargs.get("hidden_states", None)
                        if x is None and len(args) > 0:
                            x = args[0]
                    y = self.original_forward(x, *args, **kwargs)
                    return KFACAutogradFunction.apply(self, y, x)

                module.forward = new_forward.__get__(module, LoraLayer)

                weight_device = module.lora_A['default'].weight.device
                r = module.r['default']
                # BUG 1 FIX: float32 not float16.
                # float16 max=65504; large early activations silently overflow to Inf,
                # corrupting every downstream inversion and preconditioning step.
                self.a_covs[module] = torch.zeros(
                    (r, r), device=weight_device, dtype=torch.float32)
                self.g_covs[module] = torch.zeros(
                    (r, r), device=weight_device, dtype=torch.float32)
                self.optimized_modules.append(module)

                if is_vision:
                    vision_modules += 1
                else:
                    language_modules += 1

        print(f"Instrumented {vision_modules} vision modules + {language_modules} language modules")
        print(f"Total LoRA modules with K-FAC: {len(self.optimized_modules)}")

    def _compute_effective_rank(self, cov: torch.Tensor) -> float:
        try:
            with torch.no_grad():
                evals = torch.linalg.eigvalsh(cov.float())
                evals = torch.clamp(evals, min=0.0)
                total = float(evals.sum().item())
                if not math.isfinite(total) or total <= 0.0:
                    return 0.0
                p = (evals / total).clamp_min(1e-12)
                entropy = float((-(p * torch.log(p)).sum()).item())
                r_eff = math.exp(entropy)
                n = int(cov.shape[0])
                return float(max(0.0, min(n, r_eff))) if math.isfinite(r_eff) else 0.0
        except Exception:
            return 0.0

    def _get_adaptive_freq(self, base_freq, min_freq=1, max_freq=1000, window=20):
        if len(self.loss_history) < window:
            return base_freq
        recent = self.loss_history[-window:]
        first_half  = sum(recent[:window//2])  / (window//2)
        second_half = sum(recent[window//2:]) / (window//2)
        new_freq = int(base_freq * 1.5) if second_half < first_half * 0.99 \
                   else int(base_freq * 0.75)
        return max(min_freq, min(new_freq, max_freq))

    def step(self, loss=None):
        self.global_step += 1

        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.loss_history_capacity:
                self.loss_history = self.loss_history[-self.loss_history_capacity:]

        # BUG 2 FIX: gate on full window, floor at initial configured damping,
        # skip NaN/zero variance, don't fire before first inversion.
        # FIX: ceiling was hardcoded to 0.01, tuned against the old UNNORMALIZED a_cov
        # (magnitude in the thousands). Post-normalization, a_cov sits around ~16-17 on
        # the observed 500-sample run — a damping ceiling of 0.01 is ~1600x too small to
        # meaningfully regularize the K-FAC inversion, consistent with the very large
        # nat_grad_norm values (4000-6400) logged throughout that run. Now configurable via
        # config.kfac_damping_ceiling (default raised to 1.0, still tunable/testable).
        if (self.global_step >= int(getattr(self.config, 'kfac_update_freq', 50)) and
                len(self.loss_history) >= self.loss_history_capacity):
            loss_tensor = torch.tensor(self.loss_history, dtype=torch.float64)
            loss_variance = float(loss_tensor.var().item())
            if math.isfinite(loss_variance) and loss_variance > 0.0:
                adaptive = 0.001 + math.sqrt(loss_variance)
                ceiling = float(getattr(self.config, 'kfac_damping_ceiling', 1.0))
                self.config.kfac_damping = max(
                    self._initial_kfac_damping, min(ceiling, adaptive))

        kfac_freq = self._get_adaptive_freq(self.config.kfac_update_freq)
        if self.global_step - self.last_kfac_update_step >= kfac_freq:
            self.update_and_invert_factors()
            self.last_kfac_update_step = self.global_step

        reproj_warmup = int(getattr(self.config, 'reprojection_warmup_steps', 0) or 0)
        reproj_freq   = self._get_adaptive_freq(
            self.config.reprojection_freq, min_freq=1, max_freq=2000)
        if (self.global_step >= reproj_warmup and
                self.global_step - self.last_reprojection_step >= reproj_freq):
            self.neural_reprojection()
            self.last_reprojection_step = self.global_step

    def update_and_invert_factors(self):
        print(f"\nGRITManager: Inverting K-FAC factors at step {self.global_step}...")

        if not hasattr(self, "_scripted_invert_fn") or self._scripted_invert_fn is None:
            self._scripted_invert_fn = torch.jit.script(jit_invert_tensor_pair)
        scripted_invert_fn = self._scripted_invert_fn

        min_samples = getattr(self.config, 'kfac_min_samples', 256)
        inv_device  = str(getattr(self.config, 'kfac_inversion_device', 'cpu') or 'cpu')
        device_for_inv = (torch.device('cuda')
                          if inv_device == 'cuda' and torch.cuda.is_available()
                          else torch.device('cpu'))

        for module in self.optimized_modules:
            if (self.num_samples_a.get(module, 0) < min_samples or
                    self.num_samples_g.get(module, 0) < min_samples):
                continue

            a_cov = self.a_covs[module].detach().to(
                device=device_for_inv, dtype=torch.float32)
            g_cov = self.g_covs[module].detach().to(
                device=device_for_inv, dtype=torch.float32)

            a_inv, g_inv = scripted_invert_fn(
                a_cov=a_cov, g_cov=g_cov,
                kfac_damping=self.config.kfac_damping)

            if a_inv.numel() > 0 and g_inv.numel() > 0:
                self.a_invs[module] = a_inv.to('cpu')
                self.g_invs[module] = g_inv.to('cpu')

        self.factors_are_ready = True

    def log_final_effective_ranks(self):
        if not getattr(self.config, 'log_final_eff_rank', True):
            return
        print("--- Final Effective Ranks ---")
        try:
            for module in self.optimized_modules:
                r_eff_a = self._compute_effective_rank(
                    self.a_covs[module].detach().cpu().float())
                r_eff_g = self._compute_effective_rank(
                    self.g_covs[module].detach().cpu().float())
                print(f"Fisher/{module.lora_name}/eff_rank_a: {r_eff_a:.4f}")
                print(f"Fisher/{module.lora_name}/eff_rank_g: {r_eff_g:.4f}")
        except Exception:
            pass

    def log_final_raw_ranks(self):
        print("--- Final Adapted Ranks (k) ---")
        try:
            for module in self.optimized_modules:
                base_r = (int(module.r['default'])
                          if hasattr(module, 'r') and 'default' in module.r else 0)
                Va = self._last_Va_k.get(module, None)
                Vg = self._last_Vg_k.get(module, None)
                k_a = int(Va.shape[1]) if Va is not None and Va.numel() > 0 else base_r
                k_b = int(Vg.shape[1]) if Vg is not None and Vg.numel() > 0 else k_a
                print(f"Fisher/{module.lora_name}/final_k_a: {k_a}")
                print(f"Fisher/{module.lora_name}/final_k_b: {k_b}")
        except Exception:
            pass

        # Export telemetry for plot_heatmaps.py and plot_spectra.py
        try:
            run_dir = getattr(self.config, 'telemetry_dir',
                              './results_vqa_grit_llama3_2_vision')
            _export_telemetry(self, run_dir)
        except Exception as e:
            print(f"[GRIT] Telemetry export failed (non-fatal): {e}")

    def precondition_gradients(self):
        """THE PAYOFF: Modifies gradients before the optimizer step."""
        if getattr(self.config, "disable_precondition", False):
            return
        if not self.factors_are_ready:
            return

        with torch.no_grad():
            ng_warmup = int(getattr(self.config, 'ng_warmup_steps', 0) or 0)
            if self.global_step < ng_warmup:
                return

            # BUG 4 FIX: skip entire step if any module is still cold.
            min_samples = int(getattr(self.config, 'kfac_min_samples', 256) or 256)
            for module in self.optimized_modules:
                if (self.num_samples_a.get(module, 0) < min_samples or
                        self.num_samples_g.get(module, 0) < min_samples):
                    return

            _total_nat_norm_sq = torch.tensor(0.0, device=self.device)

            for module in self.optimized_modules:
                if module not in self.a_invs or module not in self.g_invs:
                    continue

                lora_a = module.lora_A['default']
                lora_b = module.lora_B['default']

                if (lora_a is None or lora_b is None
                        or lora_a.weight.grad is None
                        or lora_b.weight.grad is None):
                    continue

                grad_device = lora_a.weight.grad.device

                a_inv_f32 = self.a_invs[module].to(grad_device, dtype=torch.float32)
                g_inv_f32 = self.g_invs[module].to(grad_device, dtype=torch.float32)

                grad_b_f32 = lora_b.weight.grad.detach().to(torch.float32)
                preconditioned_b_grad_f32 = grad_b_f32 @ g_inv_f32

                grad_a_f32 = lora_a.weight.grad.detach().to(torch.float32)
                r = module.r['default']
                if r > 0:
                    grad_a_f32 = grad_a_f32.view(r, -1)
                preconditioned_a_grad_f32 = a_inv_f32 @ grad_a_f32

                _total_nat_norm_sq = (_total_nat_norm_sq
                                      + preconditioned_a_grad_f32.pow(2).sum()
                                      + preconditioned_b_grad_f32.pow(2).sum())

                preconditioned_a_grad_f32 = torch.nan_to_num(
                    preconditioned_a_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)
                preconditioned_b_grad_f32 = torch.nan_to_num(
                    preconditioned_b_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)

                # BUG 3 FIX: clamp ±10 (was ±1e6).
                # LR × clamp ≤ 0.01 → clamp ≤ 0.01/1e-4 = 100.
                # Use 10 as conservative first-run value.
                _GRAD_CLAMP = 10.0
                preconditioned_a_grad_f32 = torch.clamp(
                    preconditioned_a_grad_f32, min=-_GRAD_CLAMP, max=_GRAD_CLAMP)
                preconditioned_b_grad_f32 = torch.clamp(
                    preconditioned_b_grad_f32, min=-_GRAD_CLAMP, max=_GRAD_CLAMP)

                lora_a.weight.grad.copy_(
                    preconditioned_a_grad_f32.to(lora_a.weight.grad.dtype))
                lora_b.weight.grad.copy_(
                    preconditioned_b_grad_f32.to(lora_b.weight.grad.dtype))

                del a_inv_f32, g_inv_f32, grad_a_f32, grad_b_f32
                del preconditioned_a_grad_f32, preconditioned_b_grad_f32

            # Q6 FIX: log natural gradient norm
            nat_norm = float(_total_nat_norm_sq.sqrt().item())
            if not math.isfinite(nat_norm):
                print(f"[GRIT] WARNING: nat_grad_norm is non-finite ({nat_norm}) "
                      f"at step {self.global_step}. Check K-FAC inverses.")
            elif nat_norm > 1e4:
                print(f"[GRIT] WARNING: nat_grad_norm very large ({nat_norm:.2e}) "
                      f"at step {self.global_step}. Consider increasing kfac_damping.")
            else:
                print(f"[GRIT] nat_grad_norm={nat_norm:.4f} at step {self.global_step}")

    def neural_reprojection(self):
        """THE ADAPTATION: Prunes LoRA ranks dynamically using Fisher geometry."""
        if getattr(self.config, "disable_reprojection", False):
            return
        print(f"\nGRITManager: Neural reprojection at step {self.global_step}...")
        initial_params = 0
        final_params = 0

        with torch.no_grad():
            for module in self.optimized_modules:
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    initial_params += module.r['default'] * (
                        module.in_features + module.out_features)

            for module in self.optimized_modules:
                try:
                    lora_a = module.lora_A['default']
                    lora_b = module.lora_B['default']
                    if lora_a is None or lora_b is None:
                        continue

                    A = lora_a.weight.data.float()
                    B = lora_b.weight.data.float()
                    r = A.shape[0]
                    k = r

                    min_samples = int(
                        getattr(self.config, 'kfac_min_samples', 256) or 256)
                    if self.num_samples_a.get(module, 0) < min_samples:
                        final_params += r * (module.in_features + module.out_features)
                        continue

                    # ------------------------------------------------------------------
                    # REPROJECTION DEVICE FIX
                    #
                    # Problem: self.a_covs[module] is stored on CPU, but the EMA update
                    # in KFACAutogradFunction.backward() executes inside CUDA autocast.
                    # Under some PyTorch versions this contaminates the tensor's internal
                    # dispatch key so that torch.linalg.eigh() returns CUDA tensors even
                    # though the input looks like it is on CPU.  torch.searchsorted()
                    # then sees CPU cumulative_energy but a CUDA threshold tensor and
                    # raises: "Expected all tensors to be on the same device".
                    #
                    # Fix: .detach().cpu().float() creates a genuinely clean CPU tensor
                    # with no dispatch key contamination before eigendecomposition.
                    # device=cumulative_energy.device on the threshold makes the contract
                    # explicit instead of relying on torch.tensor() device defaults.
                    # ------------------------------------------------------------------

                    rank_adapt_start = int(
                        getattr(self.config, 'rank_adaptation_start_step', 0) or 0)

                    if (self.config.enable_rank_adaptation
                            and r > self.config.min_lora_rank
                            and self.global_step >= rank_adapt_start):

                        # A-side: force genuinely clean CPU tensor
                        cov_a_cpu = self.a_covs[module].detach().cpu().float()

                        # NaN/Inf guard on the covariance
                        if torch.isnan(cov_a_cpu).any() or torch.isinf(cov_a_cpu).any():
                            final_params += r * (module.in_features + module.out_features)
                            continue

                        evals_a, V_a = torch.linalg.eigh(cov_a_cpu)
                        order_a = torch.argsort(evals_a, descending=True)
                        V_a = V_a[:, order_a]
                        evals_a = evals_a[order_a]

                        total_energy = torch.sum(evals_a)
                        if total_energy > 1e-6:
                            cumulative_energy = (
                                torch.cumsum(evals_a, dim=0) / total_energy)
                            # device= is EXPLICIT — always CPU after cov_a_cpu
                            k_tensor = torch.searchsorted(
                                cumulative_energy,
                                torch.tensor(
                                    self.config.rank_adaptation_threshold,
                                    dtype=cumulative_energy.dtype,
                                    device=cumulative_energy.device,
                                )
                            )
                            k = int(k_tensor.item()) + 1

                        k = max(k, self.config.min_lora_rank)
                        k = min(k, r)
                        V_a_k = V_a[:, :k]

                        # G-side
                        try:
                            g_cov_m = self.g_covs.get(module, None)
                            if (getattr(self.config, 'use_two_sided_reprojection', False)
                                    and g_cov_m is not None
                                    and self.num_samples_g.get(module, 0) >= min_samples):
                                # G-side: force genuinely clean CPU tensor
                                cov_g_cpu = g_cov_m.detach().cpu().float()
                                evals_g, V_g = torch.linalg.eigh(cov_g_cpu)
                                order_g = torch.argsort(evals_g, descending=True)
                                V_g_k = V_g[:, order_g][:, :k]
                            else:
                                V_g_k = V_a_k
                        except Exception:
                            V_g_k = V_a_k

                    else:
                        k = min(self.config.reprojection_k, r)

                        # A-side: force genuinely clean CPU tensor
                        cov_a_cpu = self.a_covs[module].detach().cpu().float()
                        if torch.isnan(cov_a_cpu).any() or torch.isinf(cov_a_cpu).any():
                            final_params += r * (module.in_features + module.out_features)
                            continue

                        evals_a, V_a = torch.linalg.eigh(cov_a_cpu)
                        order_a = torch.argsort(evals_a, descending=True)
                        V_a_k = V_a[:, order_a][:, :k]

                        # G-side
                        try:
                            g_cov_m = self.g_covs.get(module, None)
                            if (getattr(self.config, 'use_two_sided_reprojection', False)
                                    and g_cov_m is not None
                                    and self.num_samples_g.get(module, 0) >= min_samples):
                                cov_g_cpu = g_cov_m.detach().cpu().float()
                                evals_g, V_g = torch.linalg.eigh(cov_g_cpu)
                                order_g = torch.argsort(evals_g, descending=True)
                                V_g_k = V_g[:, order_g][:, :k]
                            else:
                                V_g_k = V_a_k
                        except Exception:
                            V_g_k = V_a_k

                    self._last_Va_k[module] = V_a_k.detach().cpu()
                    self._last_Vg_k[module] = V_g_k.detach().cpu()

                    V_a_k_d = V_a_k.to(device=A.device, dtype=A.dtype, non_blocking=True)
                    V_g_k_d = V_g_k.to(device=B.device, dtype=A.dtype, non_blocking=True)

                    A_new = V_a_k_d @ (V_a_k_d.T @ A)
                    B_new = (B @ V_g_k_d) @ V_g_k_d.T

                    lora_a.weight.data.copy_(A_new.to(lora_a.weight.dtype))
                    lora_b.weight.data.copy_(B_new.to(lora_b.weight.dtype))

                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        final_params += k * (module.in_features + module.out_features)

                except Exception as e:
                    print(f"Error during reprojection for {module.lora_name}: {e}")
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        final_params += module.r['default'] * (
                            module.in_features + module.out_features)

        if initial_params > 0:
            reduction_pct = (initial_params - final_params) / initial_params * 100
            print(f"Neural reprojection: {initial_params:,} → {final_params:,} "
                  f"({reduction_pct:.2f}% reduction)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# === 5. JIT-Compiled Matrix Inverter ===
def jit_invert_tensor_pair(
        a_cov: torch.Tensor,
        g_cov: torch.Tensor,
        kfac_damping: float):
    with torch.no_grad():
        base_damping = float(max(kfac_damping, 1e-6))
        a_cov_f = a_cov.float(); a_cov_f = 0.5 * (a_cov_f + a_cov_f.T)
        g_cov_f = g_cov.float(); g_cov_f = 0.5 * (g_cov_f + g_cov_f.T)
        n_a = int(a_cov_f.shape[0]); n_g = int(g_cov_f.shape[0])
        dev = a_cov_f.device
        I_a = torch.eye(n_a, device=dev)
        I_g = torch.eye(n_g, device=dev)
        scales = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        for s in scales:
            L_a, info_a = torch.linalg.cholesky_ex(
                a_cov_f + float(base_damping * s) * I_a)
            L_g, info_g = torch.linalg.cholesky_ex(
                g_cov_f + float(base_damping * s) * I_g)
            if int(info_a.item()) == 0 and int(info_g.item()) == 0:
                return torch.cholesky_inverse(L_a).float(), \
                       torch.cholesky_inverse(L_g).float()
        return torch.empty(0, device=dev), torch.empty(0, device=dev)


# === 6. The GRIT Optimizer Wrapper ===
class GritOptimizer(Optimizer):

    def __init__(self, optimizer: Optimizer, grit_manager: 'GRITManager'):
        self.optimizer = optimizer
        self.grit_manager = grit_manager

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict):
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"GritOptimizer({self.optimizer.__repr__()})"

    def step(self, closure=None):
        if self.grit_manager.factors_are_ready and not getattr(
                self.grit_manager, "_preconditioned", False):
            self.grit_manager.precondition_gradients()

        max_norm = float(getattr(self.grit_manager, "_last_grad_clip_cap", 1.0))
        if max_norm > 0.0 and self.grit_manager.factors_are_ready:
            params_with_grad = []
            for module in self.grit_manager.optimized_modules:
                for key in ("lora_A", "lora_B"):
                    for sub in getattr(module, key, {}).values():
                        if hasattr(sub, "weight") and sub.weight.grad is not None:
                            params_with_grad.append(sub.weight)
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm)

        self.optimizer.step(closure)


# === 7. The GRIT Callback ===
class GritCallback(TrainerCallback):

    def __init__(self, grit_manager):
        self.grit_manager = grit_manager

    def on_step_end(self, args, state, control, **kwargs):
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        self.grit_manager.step(loss=last_loss)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.grit_manager.log_final_effective_ranks()
        except Exception:
            pass
        try:
            self.grit_manager.log_final_raw_ranks()
        except Exception:
            pass


# === 8. The GRIT Trainer ===
class GritTrainer(Seq2SeqTrainer):

    def __init__(self, processor, grit_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.grit_manager = grit_manager
        try:
            setattr(self.grit_manager, "_last_grad_clip_cap",
                    float(getattr(self.args, "max_grad_norm", 1.0)))
        except Exception:
            pass
        print("GritTrainer: Initialized with GRIT for Llama 3.2 model.")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.optimizer is not None:
            print("Wrapping the optimizer with GRIT preconditioning logic.")
            self.optimizer = GritOptimizer(self.optimizer, self.grit_manager)

    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {k: v for k, v in inputs.items() if k in self._MODEL_FORWARD_KEYS}
        outputs = model(**model_inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        lambda_k = getattr(self.grit_manager.config, "lambda_kfac", 0.0)
        lambda_r = getattr(self.grit_manager.config, "lambda_reproj", 0.0)
        if getattr(self.grit_manager.config, "disable_curvature_regularizer", False):
            lambda_k = 0.0
        if getattr(self.grit_manager.config, "disable_reprojection_regularizer", False):
            lambda_r = 0.0

        warmup_steps = int(
            getattr(self.grit_manager.config, "regularizer_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            prog = min(1.0, max(0.0,
                self.grit_manager.global_step / float(warmup_steps)))
            lambda_k = float(lambda_k) * prog
            lambda_r = float(lambda_r) * prog

        curv_reg   = torch.tensor(0.0, device=base_loss.device)
        reproj_reg = torch.tensor(0.0, device=base_loss.device)

        for module in getattr(self.grit_manager, "optimized_modules", []):
            # FIX (confirmed via smoke test): nn.ModuleDict has no .get() method (unlike
            # plain dict) — it only implements __getitem__/__contains__/keys()/items()/
            # values()/pop(). Calling .get() falls through to nn.Module.__getattr__ and
            # raises AttributeError: 'ModuleDict' object has no attribute 'get', crashing
            # on literally the first training step.
            lora_a = module.lora_A['default'] if 'default' in module.lora_A else None
            lora_b = module.lora_B['default'] if 'default' in module.lora_B else None
            if lora_a is None or lora_b is None:
                continue
            A = lora_a.weight
            B = lora_b.weight
            if A.device.type == 'meta' or B.device.type == 'meta':
                continue

            a_cov = self.grit_manager.a_covs.get(module)
            g_cov = self.grit_manager.g_covs.get(module)

            if a_cov is not None:
                A_f = A.float()
                a_cov_f = a_cov.to(device=A_f.device, dtype=A_f.dtype)
                curv_reg = curv_reg.to(A_f.device) + ((a_cov_f @ A_f) * A_f).sum()

            if g_cov is not None:
                B_f = B.float()
                g_cov_f = g_cov.to(device=B_f.device, dtype=B_f.dtype)
                curv_reg = curv_reg.to(B_f.device) + ((B_f @ g_cov_f) * B_f).sum()

            V_a_k = self.grit_manager._last_Va_k.get(module)
            V_g_k = self.grit_manager._last_Vg_k.get(module)

            if V_a_k is not None and V_a_k.numel() > 0:
                A_f = A.float(); B_f = B.float()
                device, dtype = A_f.device, A_f.dtype
                V_a = V_a_k.to(device=device, dtype=dtype, non_blocking=True)
                V_g = (V_g_k if V_g_k is not None else V_a_k).to(
                    device=device, dtype=dtype, non_blocking=True)
                P_a = V_a @ V_a.T
                P_g = V_g @ V_g.T
                I_a = torch.eye(P_a.shape[0], device=device, dtype=dtype)
                I_g = torch.eye(P_g.shape[0], device=device, dtype=dtype)
                A_res = (I_a - P_a) @ A_f
                B_res = B_f @ (I_g - P_g)
                reproj_reg = reproj_reg.to(A_res.device) + (
                    A_res.pow(2).sum() + B_res.pow(2).sum())

        loss = (base_loss
                + lambda_k * curv_reg.to(base_loss.device)
                + lambda_r * reproj_reg.to(base_loss.device))

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        # FIX (root cause confirmed via smoke test): create_optimizer_and_scheduler()'s
        # override body never executes on current transformers versions — its
        # unconditional "Wrapping the optimizer..." print never fired across a full,
        # successful run. That means self.optimizer was NEVER replaced with GritOptimizer,
        # so GritOptimizer.step()/optimizer_step() (where precondition_gradients() used to
        # be called from) were both dead code the entire time — natural-gradient
        # preconditioning never actually ran in any prior training attempt. Preconditioning
        # directly here, right after backward() and gated on sync_gradients (True only on
        # the micro-batch that's about to trigger a real optimizer step), sidesteps that
        # Trainer/Accelerate optimizer-wrapping API surface entirely.
        if (self.accelerator.sync_gradients
                and getattr(self.grit_manager, "factors_are_ready", False)):
            self.grit_manager.precondition_gradients()
            setattr(self.grit_manager, "_preconditioned", True)

        return loss.detach() / self.args.gradient_accumulation_steps

    def optimizer_step(self, *args, **kwargs):
        # NOTE: kept as a harmless redundant safety net. On transformers versions where
        # create_optimizer_and_scheduler() IS actually invoked and self.optimizer gets
        # wrapped with GritOptimizer, this would normally double-precondition against
        # training_step() above — guarded against by checking _preconditioned (set True by
        # training_step for this step already) BEFORE doing anything, not resetting it
        # first. This method itself is NOT guaranteed to be called at all — see the fix
        # note in training_step() above.
        if (getattr(self.grit_manager, "factors_are_ready", False)
                and not getattr(self.grit_manager, "_preconditioned", False)):
            self.grit_manager.precondition_gradients()
            setattr(self.grit_manager, "_preconditioned", True)
        return super().optimizer_step(*args, **kwargs)

    # Known-safe allowlist of keys MllamaForConditionalGeneration.forward() accepts. Any
    # task-specific collator (e.g. RefCOCO+'s, which also emits eval_input_ids/
    # eval_attention_mask/eval_cross_attention_mask/gt_box_norm for a future real-generation
    # eval script) can safely include extra fields in the batch without them ever reaching
    # model(**inputs) here — grit_llama.py stays task-agnostic either way.
    _MODEL_FORWARD_KEYS = {
        "input_ids", "attention_mask", "labels", "pixel_values", "image_sizes",
        "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask",
    }

    def evaluate(self, *args, **kwargs):
        print("\nClearing VRAM before evaluation...")
        gc.collect()
        torch.cuda.empty_cache()
        return super().evaluate(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        FIX (confirmed via smoke test): this used to call model.generate() whenever
        prediction_loss_only=False (i.e. whenever compute_metrics is set — which it always
        is here), returning variable-length generated token ids. metrics_refcoco.py's
        make_compute_metrics() explicitly documents that it expects same-length,
        position-aligned TEACHER-FORCED logits, never .generate() output. The mismatch
        caused a hard crash the first time evaluation ran (generated len != label len).
        Separately, the old code also passed inputs["input_ids"] — which already contains
        the teacher-forced ANSWER appended — directly into model.generate() as the prompt,
        which would have leaked the ground-truth answer into the generation context even if
        the length mismatch hadn't crashed first. Real generation-based IoU@0.5 belongs
        exclusively in evaluate_refcoco_checkpoint.py (which correctly builds a fresh
        prompt-only input); this in-training eval path must never generate.
        """
        has_labels = "labels" in inputs
        labels = inputs.get("labels") if has_labels else None

        model_inputs = {k: v for k, v in inputs.items() if k in self._MODEL_FORWARD_KEYS}

        with torch.no_grad():
            outputs = model(**model_inputs)
            loss = outputs.loss.detach() if has_labels else None
            # FIX (confirmed via real run): returning full outputs.logits here is
            # (batch, seq_len, vocab_size) — ~13GB for just 100 eval samples at Llama-3.2's
            # ~128k vocab and max_length=512. Trainer's default evaluation_loop accumulates
            # every batch's predictions on GPU before compute_metrics ever runs, causing
            # repeated CUDA OOM-retry cycles that get worse as the eval set grows (observed:
            # allocation requests climbing 6GB->26GB on a 100-sample eval). Argmax here
            # instead — metrics_refcoco.py's to_int_ids() already handles either a 3D
            # (logits) or 2D (pre-argmaxed ids) array, so this is a pure memory win with no
            # behavior change to the metric itself.
            logits = None if prediction_loss_only else outputs.logits.detach().argmax(dim=-1)

        return loss, logits, labels

