"""
ACENetClassifier: DeBERTa-v3-base with prototype context injection.

Architecture:
  DeBERTa layers 1..inject_layer produce the [CLS] hidden state, from which an
  allocation head and a temperature head produce allocation logits and a
  per-sample temperature. A Gumbel-Softmax over the allocation logits selects
  a mixture pi over K prototypes; the context vector c = pi^T D is added to all
  token hidden states at the injection layer. The upper layers then re-encode
  the input conditioned on the selected context, and a classification head
  produces the logits.

Dictionary D is built label-free (K-means over distilled contexts) and frozen:
no gradient flows through D, so label information never enters the dictionary.

Temperature: tau = tau_min + softplus(tau_head(h_cls)) * H_normalized * tau_scale,
where H_normalized = entropy(softmax(allocation_logits)) / log(K).

The forward hook stores _live_pa (allocation softmax, not detached) so that
compute_diversity_loss() can backpropagate through the allocation head; _diag
holds detached copies for logging only.
"""
import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

logger = logging.getLogger(__name__)


def _gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Gumbel-Softmax with per-sample temperature (training mode)."""
    u = torch.rand_like(logits).clamp(1e-9, 1.0 - 1e-9)
    gumbel_noise = -torch.log(-torch.log(u))
    return F.softmax((logits + gumbel_noise) / tau.clamp(0.1, 8.0), dim=-1)


def _deterministic_softmax(logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Deterministic softmax at temperature tau (inference mode, no Gumbel noise)."""
    return F.softmax(logits / tau.clamp(0.1, 8.0), dim=-1)


class ACENetClassifier(nn.Module):
    """
    ACE-Net emotion classifier: DeBERTa-v3-base + prototype context injection.

    The prototype dictionary D is frozen (label-free, K-means initialized). The
    allocation head and temperature head are trainable, and the backbone is
    unfrozen during fine-tuning. After forward(), call compute_diversity_loss()
    to obtain the differentiable diversity term.

    Args:
        backbone: HuggingFace model identifier (microsoft/deberta-v3-base)
        n_labels: number of output classes
        dict_init: numpy array (K, 768), K-means centroids (label-free)
        inject_layer: 0-indexed DeBERTa encoder layer for injection
        n_prototypes: K, must match dict_init.shape[0]
        tau_min: minimum temperature
        tau_scale: scale applied to softplus(tau_head); must be in (0, 1.0]
        use_ib_gate: if True, apply the IB gate; if False, gate is 1.0
        ib_margin: margin for the IB gate formula
        ib_w_init: initial value of the IB gate scalar weight
        is_multilabel: True for BCEWithLogitsLoss, False for CrossEntropyLoss
        cache_dir: HuggingFace model cache directory
        fixed_tau: use a constant tau = tau_min (bypass entropy-weighted tau)
        no_gumbel: deterministic softmax in training (no Gumbel noise)
        input_concat: late fusion; add c to the final [CLS] instead of injecting
        free_vector: replace the prototype mechanism with a single learnable vector
    """

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-base",
        n_labels: int = 28,
        dict_init: Optional[np.ndarray] = None,
        inject_layer: int = 8,
        n_prototypes: int = 32,
        tau_min: float = 0.3,
        tau_scale: float = 1.0,
        use_ib_gate: bool = False,
        ib_margin: float = 0.5,
        ib_w_init: float = 5.0,
        is_multilabel: bool = True,
        cache_dir: Optional[str] = None,
        fixed_tau: bool = False,
        no_gumbel: bool = False,
        input_concat: bool = False,
        free_vector: bool = False,
    ) -> None:
        super().__init__()
        assert 0.0 < tau_scale <= 1.0, (
            f"tau_scale must be in (0, 1.0]; got {tau_scale}. "
            "Values > 1.0 increase temperature (collapse risk); use 1.0 for default behavior."
        )

        self.n_labels = n_labels
        self.n_prototypes = n_prototypes
        self.inject_layer = inject_layer
        self.tau_min = tau_min
        self.tau_scale = tau_scale
        self.use_ib_gate = use_ib_gate
        self.is_multilabel = is_multilabel
        self.fixed_tau = fixed_tau
        self.no_gumbel = no_gumbel
        self.input_concat = input_concat
        self.free_vector = free_vector

        self.backbone = AutoModel.from_pretrained(backbone, cache_dir=cache_dir)
        hidden_size: int = self.backbone.config.hidden_size

        # Learnable bias vector injected at the same position as the context
        # vector; only created in free_vector mode and bypasses all other modules.
        if free_vector:
            self.v_free = nn.Parameter(torch.zeros(hidden_size))

        # Allocation head: h_cls -> K allocation logits.
        self.allocation_head = nn.Linear(hidden_size, n_prototypes, bias=True)
        nn.init.xavier_uniform_(self.allocation_head.weight)
        nn.init.zeros_(self.allocation_head.bias)

        # Temperature head: h_cls -> 1 scalar.
        self.temperature_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.temperature_head.bias)

        if use_ib_gate:
            self.w_ib = nn.Parameter(torch.tensor(float(ib_w_init)))
            self.register_buffer("m_ib", torch.tensor(float(ib_margin)))

        # Prototype dictionary D (label-free, frozen). register_buffer keeps it
        # in the state_dict and moves it with .to(device) without a gradient.
        if dict_init is not None:
            assert dict_init.shape == (n_prototypes, hidden_size), (
                f"dict_init shape {dict_init.shape} != (K={n_prototypes}, d={hidden_size})"
            )
            d_tensor = torch.tensor(dict_init, dtype=torch.float32)
        else:
            logger.warning(
                "[ACENetClassifier] dict_init=None; random init (tests only, not for experiments)"
            )
            d_tensor = torch.randn(n_prototypes, hidden_size)
        self.register_buffer("prototype_dict", d_tensor)

        self.classifier = nn.Linear(hidden_size, n_labels)

        # Per-forward-pass hook state.
        self._current_mask: Optional[torch.Tensor] = None
        # _live_pa: live (non-detached) allocation softmax for the diversity gradient.
        self._live_pa: Optional[torch.Tensor] = None
        # _diag: detached diagnostic tensors for logging.
        self._diag: Dict[str, torch.Tensor] = {}
        # _live_c: detached context vector from the most recent forward pass.
        self._live_c: Optional[torch.Tensor] = None
        # _swap_context: externally injected context (None = normal allocation mode).
        self._swap_context: Optional[torch.Tensor] = None
        # _last_c: context stored for late fusion in input_concat mode.
        self._last_c: Optional[torch.Tensor] = None

        self._hook_handle = self.backbone.encoder.layer[inject_layer].register_forward_hook(
            self._layer_hook
        )

        logger.info(
            f"[ACENetClassifier] backbone={backbone} | inject_layer={inject_layer} "
            f"| K={n_prototypes} | tau_min={tau_min} | tau_scale={tau_scale} "
            f"| gate={'ON' if use_ib_gate else 'OFF'} | n_labels={n_labels} "
            f"| multilabel={is_multilabel} | fixed_tau={fixed_tau} | no_gumbel={no_gumbel} "
            f"| input_concat={input_concat} | free_vector={free_vector}"
        )

    def _layer_hook(
        self,
        module: nn.Module,
        inputs: Tuple,
        output,
    ):
        """
        Forward hook on encoder.layer[inject_layer].

        Computes the context vector c from the prototype dictionary and adds it
        to all token hidden states so that the upper layers re-encode the input
        conditioned on the selected context. Entropy, log, and Gumbel-Softmax
        run in fp32 (autocast disabled) for numerical stability; c is cast back
        to the original dtype before injection.
        """
        hidden = output[0] if isinstance(output, tuple) else output

        mask = self._current_mask
        if mask is None:
            logger.warning("[hook] attention_mask is None; injection skipped")
            return output

        orig_dtype = hidden.dtype

        # free_vector mode: bypass the entire prototype mechanism.
        if self.free_vector:
            with torch.cuda.amp.autocast(enabled=False):
                c = self.v_free.float().unsqueeze(0).expand(hidden.shape[0], -1)
            c_out = c.to(orig_dtype)
            mask_float = mask.unsqueeze(-1).to(orig_dtype)
            c_broadcast = c_out.unsqueeze(1) * mask_float
            new_hidden = hidden + c_broadcast
            self._live_pa = torch.full(
                (hidden.shape[0], self.n_prototypes), 1.0 / self.n_prototypes,
                device=hidden.device, dtype=torch.float32,
            )
            self._diag = {}
            self._live_c = None
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        # fp32 zone: all hook computations forced out of autocast.
        with torch.cuda.amp.autocast(enabled=False):
            cls_h = hidden[:, 0, :].float()

            alloc_logits = self.allocation_head(cls_h)

            pa = F.softmax(alloc_logits, dim=-1)
            if self.fixed_tau:
                H_norm = torch.zeros(cls_h.shape[0], 1, device=cls_h.device)
                tau = torch.full(
                    (cls_h.shape[0], 1), self.tau_min,
                    device=cls_h.device, dtype=cls_h.dtype,
                )
            else:
                H = -(pa * pa.clamp(min=1e-9).log()).sum(dim=-1, keepdim=True)
                H_norm = H / math.log(self.n_prototypes)
                tau = self.tau_min + F.softplus(
                    self.temperature_head(cls_h)
                ) * H_norm * self.tau_scale

            if self.training and not self.no_gumbel:
                pi = _gumbel_softmax(alloc_logits, tau)
            else:
                pi = _deterministic_softmax(alloc_logits, tau)

            # Use an externally injected context in swap mode, otherwise compute
            # it from the frozen dictionary (no gradient flows through D).
            if self._swap_context is not None:
                c = self._swap_context.float()
            else:
                c = pi @ self.prototype_dict.float()
            self._live_c = c.detach().clone()

            if self.use_ib_gate:
                cos_sim = F.cosine_similarity(c, cls_h, dim=-1, eps=1e-8)
                g_ib = torch.sigmoid(self.w_ib * (self.m_ib - cos_sim))
                c = c * g_ib.unsqueeze(-1)

        # Cast c back so the layers after the hook keep their original precision.
        c_out = c.to(orig_dtype)

        if self.input_concat:
            self._last_c = c_out
            new_hidden = hidden
        else:
            mask_float = mask.unsqueeze(-1).to(orig_dtype)
            c_broadcast = c_out.unsqueeze(1) * mask_float
            new_hidden = hidden + c_broadcast

        # _live_pa: non-detached for the diversity-loss gradient.
        self._live_pa = pa

        self._diag = {
            "pa": pa.detach(),
            "tau": tau.detach(),
            "H_norm": H_norm.detach(),
            "pi": pi.detach(),
        }

        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single forward pass. The hook on encoder.layer[inject_layer] fires
        during backbone() and injects context into the hidden states.

        Returns raw logits (B, n_labels); apply sigmoid+threshold (multi-label)
        or argmax (multi-class) for prediction.
        """
        self._current_mask = attention_mask
        self._last_c = None

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_final = out.last_hidden_state[:, 0, :]

        if self.input_concat and self._last_c is not None:
            cls_final = cls_final + self._last_c

        logits = self.classifier(cls_final)

        self._current_mask = None
        return logits

    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Differentiable diversity regularization loss. Call AFTER forward().

        diversity_loss = sample_entropy - batch_entropy, where sample_entropy is
        E_batch[H(pi_sample)] (minimized, sharpens per-sample allocation) and
        batch_entropy is H(E_batch[pi]) (maximized, encourages uniform prototype
        usage). Both terms are normalized by log(K). The caller weights this by
        the diversity weight before adding it to the total loss.

        Raises:
            RuntimeError: if called before forward()
        """
        if self._live_pa is None:
            raise RuntimeError(
                "compute_diversity_loss() must be called after forward(). "
                "No _live_pa available (forward() not yet called for this batch)."
            )

        pa = self._live_pa
        K = self.n_prototypes

        mp = pa.mean(dim=0)
        be = -(mp * mp.clamp(min=1e-9).log()).sum() / math.log(K)

        se = -(pa * pa.clamp(min=1e-9).log()).sum(dim=-1).mean() / math.log(K)

        return se - be

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        """
        Return detached diagnostic tensors from the most recent forward pass
        (keys 'pa', 'tau', 'H_norm', 'pi'); empty dict before the first forward().
        """
        return self._diag

    def remove_hook(self) -> None:
        """Remove the forward hook. Call after training if the model is reused."""
        self._hook_handle.remove()
        logger.info(f"[ACENetClassifier] Forward hook removed from layer {self.inject_layer}")

    def assert_no_label_gradient_in_dict(self) -> None:
        """Assert prototype_dict.requires_grad is False (label-free invariant)."""
        assert not self.prototype_dict.requires_grad, (
            "INVARIANT VIOLATION: prototype_dict.requires_grad must be False. "
            "Label gradients must never reach the dictionary. "
            "Check that dict_init was set as register_buffer (not nn.Parameter)."
        )

    def assert_pi_on_simplex(self, pi: torch.Tensor, atol: float = 1e-4) -> None:
        """Assert the prototype weights sum to 1 (simplex constraint)."""
        diff = (pi.sum(dim=-1) - 1.0).abs()
        assert diff.max().item() < atol, (
            f"pi sums deviate from 1.0 by {diff.max().item():.6f} (atol={atol}). "
            "Gumbel-Softmax output must lie on the K-simplex."
        )

    def assert_ib_gate_in_range(self, g_ib: torch.Tensor, atol: float = 1e-6) -> None:
        """Assert IB gate values are in [0, 1]."""
        assert (g_ib >= -atol).all() and (g_ib <= 1.0 + atol).all(), (
            f"IB gate values out of range [0, 1]: min={g_ib.min()}, max={g_ib.max()}"
        )

    def set_swap_context(self, c_swap: torch.Tensor) -> None:
        """
        Set a forced context for the swap-contrastive pass. Call before the
        forward-swap pass; the hook uses c_swap instead of the normal allocation.
        """
        self._swap_context = c_swap

    def clear_swap_context(self) -> None:
        """Clear the forced context and return to normal allocation mode."""
        self._swap_context = None

    def get_live_c(self) -> Optional[torch.Tensor]:
        """
        Return the context vector from the most recent forward pass (detached
        fp32), or None if no forward pass has run.
        """
        return self._live_c
