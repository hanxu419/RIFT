# RIFT model implementation.
# Implementation provenance: docs/provenance.md.
# Licensing: ThirdPartyNotices.txt.

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import similarity, KgeLoss, rat
from kge.protocol import validate_protocol_config

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter

try:
    from pytorch_pretrained_bert.modeling import (
        BertEncoder,
        BertConfig,
        BertLayerNorm,
        BertPreTrainedModel,
    )
    LEGACY_BERT_AVAILABLE = True
except ImportError:
    LEGACY_BERT_AVAILABLE = False
    class BertConfig:
        def __init__(self, *args, **kwargs):
            self.hidden_size = kwargs.get("hidden_size", 256)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 2)
            self.num_attention_heads = kwargs.get("num_attention_heads", 8)
            self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
            self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
            self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
            self.initializer_range = kwargs.get("initializer_range", 0.02)

    class BertLayerNorm(nn.LayerNorm):
        def __init__(self, hidden_size, eps=1e-12):
            super().__init__(hidden_size, eps=eps)

    class BertPreTrainedModel:
        @staticmethod
        def init_bert_weights(_self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    class BertEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=int(config.hidden_size),
                nhead=int(config.num_attention_heads),
                dim_feedforward=int(config.intermediate_size),
                dropout=float(config.hidden_dropout_prob),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(config.num_hidden_layers),
            )

        def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
            # The legacy call site uses a full-ones mask, so this fallback can ignore it.
            out = self.encoder(hidden_states)
            if output_all_encoded_layers:
                return [out]
            return (out,)

from functools import partial

import math

from kge.util import sc

from kge.util import mixer2
from kge.model.causal_context import (
    select_past_context,
    select_dual_past_context,
    select_time_stratified_past_context,
)



class RIFTTimeEvolution(nn.Module):
    """CTE: a timestamp-conditioned basis-shared block transform for RIFT.

    Share K basis block-matrices and condition their mixture on time embeddings.
    This implementation is preserved; normalization does not change its equations.

    For each time embedding u_t, we predict mixture weights pi_t (softmax) and build
    a residual transform per token:
        X_t = I + scale * sum_k pi_t[k] * B_k
    Then apply X_t blockwise to the entity embedding.

    Notes:
      - Designed to be GPU-friendly (einsum-based, no Python loops).
      - Regularization is returned as a scalar (raw L2); weighting is applied outside.
    """

    def __init__(
        self,
        dim: int,
        basis_k: int = 8,
        block_size: int = 16,
        coeff_mlp_hidden: int = 128,
        coeff_mlp_layers: int = 2,
        coeff_temperature: float = 1.0,
        activation: str = "gelu",
        residual: bool = True,
        diagonal_only: bool = False,
        scale_init: float = 0.0,
        basis_init_std: float = 0.02,
    ):
        super().__init__()
        if dim % block_size != 0:
            raise ValueError(f"RIFTTimeEvolution: dim={dim} must be divisible by block_size={block_size}")
        if basis_k <= 0:
            raise ValueError("RIFTTimeEvolution: basis_k must be > 0")
        if coeff_temperature <= 0:
            raise ValueError("RIFTTimeEvolution: coeff_temperature must be > 0")

        self.dim = int(dim)
        self.basis_k = int(basis_k)
        self.block_size = int(block_size)
        self.block_num = self.dim // self.block_size
        self.coeff_temperature = float(coeff_temperature)
        self.residual = bool(residual)
        self.diagonal_only = bool(diagonal_only)

        # Shared basis: [K, B, b, b]
        self.basis = nn.Parameter(torch.zeros(self.basis_k, self.block_num, self.block_size, self.block_size))
        nn.init.normal_(self.basis, mean=0.0, std=float(basis_init_std))

        # Optional: keep only diagonal entries within each block
        if self.diagonal_only:
            eye = torch.eye(self.block_size).view(1, 1, self.block_size, self.block_size)
            self.register_buffer("_diag_mask", eye)
        else:
            self._diag_mask = None

        act = activation.lower()
        if act == "relu":
            act_layer = nn.ReLU
        elif act == "gelu":
            act_layer = nn.GELU
        elif act == "tanh":
            act_layer = nn.Tanh
        else:
            raise ValueError(f"RIFTTimeEvolution: unsupported activation '{activation}' (use relu/gelu/tanh)")

        hidden = int(coeff_mlp_hidden)
        n_layers = int(coeff_mlp_layers)
        layers = []
        if n_layers <= 1:
            layers.append(nn.Linear(self.dim, self.basis_k))
        else:
            layers.append(nn.Linear(self.dim, hidden))
            layers.append(act_layer())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(act_layer())
            layers.append(nn.Linear(hidden, self.basis_k))
        self.coeff_mlp = nn.Sequential(*layers)

        # Residual scale (learnable). scale_init=0 => starts as identity map.
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, ent_emb: Tensor, time_emb: Tensor):
        """Return (evolved_ent_emb, reg_l2_raw)."""
        if ent_emb.numel() == 0:
            return ent_emb, ent_emb.new_zeros(())

        if ent_emb.shape != time_emb.shape:
            raise ValueError(f"RIFTTimeEvolution: time_emb shape {tuple(time_emb.shape)} must match ent_emb {tuple(ent_emb.shape)}")
        if ent_emb.shape[-1] != self.dim:
            raise ValueError(f"RIFTTimeEvolution: last dim mismatch: got {ent_emb.shape[-1]} expected {self.dim}")

        ent_shape = ent_emb.shape
        ent = ent_emb.reshape(-1, self.dim)
        tim = time_emb.reshape(-1, self.dim)

        # Mixture weights: [N, K]
        logits = self.coeff_mlp(tim) / self.coeff_temperature
        pi = torch.softmax(logits, dim=-1)

        basis = self.basis
        if self._diag_mask is not None:
            basis = basis * self._diag_mask

        # Combine basis into per-token block matrices: [N, B, b, b]
        M = torch.einsum("nk,kbxy->nbxy", pi, basis)

        # Residual transform
        if self.residual:
            I = torch.eye(self.block_size, device=ent.device, dtype=ent.dtype).view(1, 1, self.block_size, self.block_size)
            M = I + self.scale * M
        else:
            M = self.scale * M

        ent_blocks = ent.view(-1, self.block_num, self.block_size)
        out_blocks = torch.einsum("nbx,nbxy->nby", ent_blocks, M)
        out = out_blocks.reshape(ent_shape)

        # Raw L2 regularization term; weighting is applied by the caller
        reg = self.basis.pow(2).mean() + self.scale.pow(2)
        reg = reg.to(ent_emb.dtype)
        return out, reg


class RIFTScorer(RelationalScorer):
    r"""Implementation of the RIFT KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        validate_protocol_config(config, where="RIFT model construction")
        if self.get_option("require_legacy_bert") and not LEGACY_BERT_AVAILABLE:
            raise RuntimeError(
                "This RIFT config requires pytorch-pretrained-bert==0.6.0; "
                "install the project dependencies before training."
            )
        self.dim = self.get_option("entity_embedder.dim")
        self.max_context_size = self.get_option("max_context_size")
        self.initializer_range = self.get_option("initializer_range")

        self.cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.cls, std=self.initializer_range)
        self.global_cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.global_cls, std=self.initializer_range)
        self.local_mask = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.local_mask, std=self.initializer_range)
        self.type_embeds = nn.Embedding(100, self.dim)
        torch.nn.init.normal_(self.type_embeds.weight, std=self.initializer_range)
        self.atomic_type_embeds = nn.Embedding(4, self.dim)
        torch.nn.init.normal_(self.atomic_type_embeds.weight, std=self.initializer_range)

        self.similarity = getattr(similarity, self.get_option("similarity"))(self.dim)
        self.layer_norm = BertLayerNorm(self.dim, eps=1e-12)
        self.atomic_layer_norm = BertLayerNorm(self.dim, eps=1e-12)
        
        self.mixer_encoder = mixer2.MLPMixer(
            # num_channel=3,
            num_ctx=self.max_context_size + 1,
            dim = self.dim,
            depth=self.get_option("nlayer"),
            ctx_dim=self.get_option("ff_dim"),
            # channel_dim=self.get_option("ff_dim"),
            dropout=self.get_option("attn_dropout")
        )
        
        self.glb_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # ------------------------------
        # Stage-2: RCAP / GCFF / CDC
        # ------------------------------
        # Neighbor ordering is handled per-batch in _get_encoder_output

        # RCAP: relation-conditioned attention pooling over (S + neighbors)
        self.dra_enable = bool(self.get_option("rcap.enable"))
        if self.dra_enable:
            self.dra_key_dim = int(self.get_option("rcap.key_dim"))
            self.dra_q = nn.Linear(self.dim, self.dra_key_dim, bias=False)
            self.dra_k = nn.Linear(self.dim, self.dra_key_dim, bias=False)
            self.dra_attn_dropout = nn.Dropout(float(self.get_option("rcap.attn_dropout")))
        else:
            self.dra_key_dim = 0
            self.dra_q = None
            self.dra_k = None
            self.dra_attn_dropout = None

        # GCFF: gated conditional feature fusion before scoring
        self.gltf_enable = bool(self.get_option("gcff.enable"))
        if self.gltf_enable:
            self.gltf_use_relation = bool(self.get_option("gcff.use_relation"))
            self.gltf_global_proj = nn.Linear(self.dim, self.dim, bias=False) if bool(self.get_option("gcff.global_proj")) else None
            gate_in_dim = self.dim * (4 if self.gltf_use_relation else 3)  # [local, global, (rel), time]
            gate_hidden = int(self.get_option("gcff.gate_hidden"))
            self.gltf_gate = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(float(self.get_option("gcff.gate_dropout"))),
                nn.Linear(gate_hidden, self.dim),
            )
            # Initialize gate to ~0.5 (safe start)
            nn.init.zeros_(self.gltf_gate[-1].weight)
            nn.init.zeros_(self.gltf_gate[-1].bias)
        else:
            self.gltf_use_relation = False
            self.gltf_global_proj = None
            self.gltf_gate = None

        # CDC: counterfactual debias contrastive loss
        self.cdc_enable = bool(self.get_option("cdc.enable"))
        self.cdc_tau = float(self.get_option("cdc.tau")) if self.cdc_enable else 0.07
        self.cdc_subsample_ratio = float(self.get_option("cdc.subsample_ratio")) if self.cdc_enable else 0.0
        self.cdc_shuffle_relation = bool(self.get_option("cdc.shuffle_relation")) if self.cdc_enable else False
        self.cdc_shuffle_time = bool(self.get_option("cdc.shuffle_time")) if self.cdc_enable else False

        # ------------------------------
        # Neighbor index device cache (performance)
        # ------------------------------
        # The dataset neighbor index (ctx_list/ctx_size) is static. If it lives on CPU, the
        # baseline code will (1) copy ids to CPU for index_select, then (2) copy ctx_ids back
        # to GPU every batch, which creates H2D stalls and "gear-shaped" GPU utilization.
        #
        # We cache a device copy once:
        #   - First call on a CUDA device: move (ctx_list, ctx_size) to that CUDA device once.
        #   - Subsequent batches reuse the cached tensors directly on GPU.
        # This is safe because the neighbor index is immutable.
        self._neighbor_index_cpu = None          # (ctx_list, ctx_size) as returned by dataset.index("neighbor")
        self._neighbor_index_cache = {}          # str(device) -> (ctx_list, ctx_size) on that device
        self.neighbor_causal = bool(self.get_option("neighbor.causal"))
        # The explicit policy is saved with the checkpoint. A legacy
        # dual_history=true config still selects the same relation-based path.
        legacy_dual_history = (
            bool(self.get_option("neighbor.dual_history"))
            if self.has_option("neighbor.dual_history") else False
        )
        retrieval_policy = (
            str(self.get_option("neighbor.retrieval_policy")).lower()
            if self.has_option("neighbor.retrieval_policy") else "latest32"
        )
        allowed_policies = ("latest32", "relation_16_16", "time_stratified_16_16")
        if retrieval_policy not in allowed_policies:
            raise ValueError(f"neighbor.retrieval_policy must be one of {allowed_policies}")
        if legacy_dual_history:
            if retrieval_policy == "latest32":
                retrieval_policy = "relation_16_16"
            elif retrieval_policy != "relation_16_16":
                raise ValueError("neighbor.dual_history conflicts with neighbor.retrieval_policy")
        self.neighbor_retrieval_policy = retrieval_policy
        self.neighbor_dual_history = retrieval_policy == "relation_16_16"
        self.neighbor_time_stratified_history = retrieval_policy == "time_stratified_16_16"
        if retrieval_policy != "latest32":
            if not self.neighbor_causal:
                raise ValueError("non-latest retrieval requires neighbor.causal: true")
            if self.max_context_size != 32:
                raise ValueError("non-latest retrieval requires max_context_size: 32")

        # self.transformer_encoder = rat.Encoder(
        #     lambda: rat.EncoderLayer(
        #         self.dim,
        #         rat.MultiHeadedAttentionWithRelations(
        #             self.get_option("nhead"),
        #             self.dim,
        #             self.get_option("attn_dropout")),
        #         rat.PositionwiseFeedForward(
        #             self.dim,
        #             self.get_option("ff_dim"),
        #             self.get_option("hidden_dropout")),
        #         num_relation_kinds=0,
        #         dropout=self.get_option("hidden_dropout")),
        #     self.get_option("nlayer"),
        #     self.initializer_range,
        #     tie_layers=False)

        config = BertConfig(0, hidden_size=self.dim,
                            num_hidden_layers= self.get_option("nlayer") // 2, # self.get_option("nlayer") // 2, # // 2
                            num_attention_heads=self.get_option("nhead"),
                            intermediate_size=self.get_option("ff_dim"),
                            hidden_act=self.get_option("activation"),
                            hidden_dropout_prob=self.get_option("hidden_dropout"),
                            attention_probs_dropout_prob=self.get_option("attn_dropout"),
                            max_position_embeddings=0,  # no effect
                            type_vocab_size=0,  # no effect
                            initializer_range=self.initializer_range)
        self.atom_encoder = BertEncoder(config)
        self.atom_encoder.config = config
        self.atom_encoder.apply(partial(BertPreTrainedModel.init_bert_weights, self.atom_encoder))

        # ------------------------------
        # CTE: timestamp-conditioned transform
        # ------------------------------
        self.time_evolver = None
        self.time_evolver_enable = self.get_option("cte.enable")
        if self.time_evolver_enable:
            evolver_type = self.get_option("cte.type")
            if evolver_type != "basis_block":
                raise ValueError(f"rift.cte.type must be 'basis_block', got '{evolver_type}'")
            self.time_evolver = RIFTTimeEvolution(
                dim=self.dim,
                basis_k=self.get_option("cte.basis_k"),
                block_size=self.get_option("cte.block_size"),
                coeff_mlp_hidden=self.get_option("cte.coeff_mlp_hidden"),
                coeff_mlp_layers=self.get_option("cte.coeff_mlp_layers"),
                coeff_temperature=self.get_option("cte.coeff_temperature"),
                activation=self.get_option("cte.activation"),
                residual=self.get_option("cte.residual"),
                diagonal_only=self.get_option("cte.diagonal_only"),
                scale_init=self.get_option("cte.scale_init"),
                basis_init_std=self.get_option("cte.basis_init_std"),
            )
        # ------------------------------
        # Relative Δt Embedding (long-span "how long ago", Stage-2+)
        # ------------------------------
        self.rel_dt_enable = bool(self.get_option("rel_dt.enable"))
        if self.rel_dt_enable:
            self.rel_dt_num_buckets = int(self.get_option("rel_dt.num_buckets"))
            self.rel_dt_exact_buckets = int(self.get_option("rel_dt.exact_buckets"))
            self.rel_dt_mode = str(self.get_option("rel_dt.mode")).lower()
            max_dt_cfg = int(self.get_option("rel_dt.max_dt"))
            if max_dt_cfg <= 0:
                # Infer from dataset if possible; fall back to a conservative constant
                if hasattr(self.dataset, "num_times") and callable(getattr(self.dataset, "num_times")):
                    max_dt_cfg = int(self.dataset.num_times()) - 1
                else:
                    max_dt_cfg = 1000
            self.rel_dt_max_dt = max(1, max_dt_cfg)

            self.rel_dt_embed = nn.Embedding(self.rel_dt_num_buckets, self.dim)
            torch.nn.init.normal_(self.rel_dt_embed.weight, std=self.initializer_range)
            self.rel_dt_dropout = nn.Dropout(float(self.get_option("rel_dt.dropout")))
            # Learnable scalar gate; 0.0 keeps baseline behavior initially
            self.rel_dt_scale = nn.Parameter(torch.tensor(float(self.get_option("rel_dt.scale_init"))))
        else:
            self.rel_dt_num_buckets = 0
            self.rel_dt_exact_buckets = 0
            self.rel_dt_mode = "off"
            self.rel_dt_max_dt = 0
            self.rel_dt_embed = None
            self.rel_dt_dropout = None
            self.rel_dt_scale = None


        # ------------------------------
        # Stage-3: HEPE (historical entity prior enhancement)
        # ------------------------------
        self.history_copy_enable = bool(self.get_option("hepe.enable"))
        self.history_copy_mode = str(self.get_option("hepe.mode")).lower()
        if self.history_copy_mode not in ("fixed", "adaptive"):
            raise ValueError("hepe.mode must be 'fixed' or 'adaptive'")
        self.history_attention_mix = float(self.get_option("hepe.attention_mix"))
        if not 0.0 <= self.history_attention_mix <= 1.0:
            raise ValueError("hepe.attention_mix must be in [0, 1]")
        self.history_gate_enabled = bool(self.get_option("hepe.gate_enabled"))
        self.history_gate_time_feature = str(self.get_option("hepe.gate_time_feature")).lower()
        if self.history_gate_time_feature not in ("past_age", "none"):
            raise ValueError("hepe.gate_time_feature must be 'past_age' or 'none'")
        self.history_gate = None
        if self.history_copy_enable:
            self.history_copy_weight = float(self.get_option("hepe.weight"))
            self.history_copy_normalize = str(self.get_option("hepe.normalize")).lower()
            self.history_copy_time_gamma = float(self.get_option("hepe.time_gamma"))
            self.history_copy_apply_in_training = bool(self.get_option("hepe.apply_in_training"))
            if self.history_copy_mode == "adaptive":
                if not self.neighbor_causal and self.history_gate_time_feature != "none":
                    raise ValueError("Noncausal adaptive history copy requires gate_time_feature: none")
                if not self.neighbor_causal and self.history_copy_time_gamma != 1.0:
                    raise ValueError("Noncausal adaptive history copy requires time_gamma: 1.0")
                if self.history_gate_enabled:
                    gate_hidden = int(self.get_option("hepe.gate_hidden"))
                    self.history_gate = nn.Sequential(
                        nn.Linear(2 * self.dim + 3, gate_hidden),
                        nn.GELU(),
                        nn.Linear(gate_hidden, 1),
                    )
                    # 2*sigmoid(0)=1: the initial gate matches fixed HEPE scale.
                    nn.init.zeros_(self.history_gate[-1].weight)
                    nn.init.zeros_(self.history_gate[-1].bias)
        else:
            self.history_copy_weight = 0.0
            self.history_copy_normalize = "mean"
            self.history_copy_time_gamma = 1.0
            self.history_copy_apply_in_training = False


    def _bucketize_rel_dt(self, dt_ids: Tensor) -> Tensor:
        """Bucketize non-negative Δt = (query_time_id - neighbor_time_id) into discrete bins.

        - dt_ids: LongTensor of shape [bs, L] (or any shape), must be >= 0 (we clamp defensively).
        - returns: LongTensor with same shape in [0, num_buckets-1]
        """
        if dt_ids.numel() == 0:
            return dt_ids.new_zeros(dt_ids.shape, dtype=torch.long)

        dt = dt_ids
        if dt.dtype != torch.long:
            dt = dt.long()
        dt = dt.clamp(min=0)

        num_buckets = int(getattr(self, "rel_dt_num_buckets", 0))
        if (not getattr(self, "rel_dt_enable", False)) or num_buckets <= 0:
            return dt.new_zeros(dt.shape, dtype=torch.long)

        mode = str(getattr(self, "rel_dt_mode", "log")).lower()
        if mode in ("linear", "lin"):
            return dt.clamp(max=num_buckets - 1)

        # Log-scale bucketing: keep small dt exact, compress large dt.
        exact = int(getattr(self, "rel_dt_exact_buckets", 32))
        exact = max(1, min(exact, num_buckets))
        max_dt = int(getattr(self, "rel_dt_max_dt", 1000))
        if max_dt <= exact:
            return dt.clamp(max=num_buckets - 1)

        dt = dt.clamp(max=max_dt)

        # bucket = exact + floor( log(dt/exact) / log(max_dt/exact) * (num_buckets-exact) ), for dt >= exact
        dt_f = dt.float()
        log_ratio = torch.log(dt_f / float(exact)) / math.log(float(max_dt) / float(exact))
        log_bucket = (log_ratio * float(num_buckets - exact)).long()
        buckets = torch.where(dt < exact, dt, exact + log_bucket)
        return buckets.clamp(min=0, max=num_buckets - 1)

    # ------------------------------
    # Stage-2 helpers
    # ------------------------------
    def _fuse_gltf(self, local_repr: Tensor, global_repr: Tensor, p_emb: Tensor, t_emb: Tensor) -> Tensor:
        """GCFF: gated conditional fusion of local context representation and global entity embedding."""
        if not getattr(self, "gltf_enable", False) or self.gltf_gate is None:
            return local_repr
        g = global_repr
        if self.gltf_global_proj is not None:
            g = self.gltf_global_proj(g)
        pieces = [local_repr, g]
        if getattr(self, "gltf_use_relation", False):
            pieces.append(p_emb)
        pieces.append(t_emb)
        gate_in = torch.cat(pieces, dim=-1)
        gate = torch.sigmoid(self.gltf_gate(gate_in))
        fused = gate * local_repr + (1.0 - gate) * g
        fused = F.dropout(fused, p=float(self.get_option("gcff.dropout")), training=self.training)
        return fused

    def _cdc_nt_xent(self, z_a: Tensor, z_pos: Tensor, z_cf: Tensor, tau: float) -> Tensor:
        """NT-Xent / InfoNCE with additional counterfactual negatives."""
        if z_a.numel() == 0:
            return z_a.new_zeros(())
        z_a = F.normalize(z_a, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        z_cf = F.normalize(z_cf, dim=-1)
        logits_pos = (z_a @ z_pos.t()) / tau  # [m, m]
        logits_cf = (z_a @ z_cf.t()) / tau    # [m, m]
        logits = torch.cat([logits_pos, logits_cf], dim=1)  # [m, 2m]
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, labels)

    def _maybe_compute_cdc_loss(
        self,
        z1_full: Tensor,
        s_emb: Tensor,
        p_emb: Tensor,
        t_emb: Tensor,
        ids: Tensor,
        gt_ent: Tensor,
        gt_rel: Tensor,
        gt_tim: Tensor,
        t_ids: Tensor,
        history_relation_ids: Tensor = None,
    ) -> Tensor:
        """Compute CDC loss on a subsampled slice of the batch."""
        if not getattr(self, "cdc_enable", False) or self.cdc_subsample_ratio <= 0:
            return z1_full.new_zeros(())
        device = z1_full.device
        n = z1_full.size(0)
        # subsample
        sel = torch.rand(n, device=device) < float(self.cdc_subsample_ratio)
        idx = sel.nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            return z1_full.new_zeros(())
        # View-1 (from current forward)
        z1 = z1_full.index_select(0, idx)

        # View-2 (dropout-augmented)
        s2 = s_emb.index_select(0, idx)
        p2 = p_emb.index_select(0, idx)
        t2 = t_emb.index_select(0, idx)
        ids2 = ids.index_select(0, idx)
        ge2 = gt_ent.index_select(0, idx)
        gr2 = gt_rel.index_select(0, idx)
        gt2 = gt_tim.index_select(0, idx)
        tid2 = t_ids.index_select(0, idx)
        hr2 = history_relation_ids.index_select(0, idx) if history_relation_ids is not None else None

        local2, _, _ = self._get_encoder_output(
            s2, p2, t2, ids2, ge2, gr2, gt2, tid2, history_relation_ids=hr2
        )
        z2 = self._fuse_gltf(local2, s2, p2, t2)

        # Counterfactual branch: shuffle relation/time (optionally)
        p_cf = p2
        t_cf = t2
        gr_cf = gr2
        gt_cf = gt2
        tid_cf = tid2
        hr_cf = hr2

        m = idx.numel()
        if getattr(self, "cdc_shuffle_relation", False):
            perm_r = torch.randperm(m, device=device)
            p_cf = p2.index_select(0, perm_r)
            gr_cf = gr2.index_select(0, perm_r)
            if hr2 is not None:
                hr_cf = hr2.index_select(0, perm_r)
        if getattr(self, "cdc_shuffle_time", False):
            perm_t = torch.randperm(m, device=device)
            t_cf = t2.index_select(0, perm_t)
            gt_cf = gt2.index_select(0, perm_t)
            tid_cf = tid2.index_select(0, perm_t)

        local_cf, _, _ = self._get_encoder_output(
            s2, p_cf, t_cf, ids2, ge2, gr_cf, gt_cf, tid_cf,
            history_relation_ids=hr_cf,
        )
        zcf = self._fuse_gltf(local_cf, s2, p_cf, t_cf)

        tau = float(self.cdc_tau)
        loss_a = self._cdc_nt_xent(z1, z2, zcf, tau)
        loss_b = self._cdc_nt_xent(z2, z1, zcf, tau)
        return 0.5 * (loss_a + loss_b)

    def _get_neighbor_index(self, device: torch.device):
        """Cache the selected immutable neighbor index on the requested device."""
        dev = torch.device(device)
        key = str(dev)
        cached = self._neighbor_index_cache.get(key, None)
        if cached is not None:
            return cached

        # Load once from dataset (usually on CPU)
        if self._neighbor_index_cpu is None:
            index_name = (
                "causal_neighbor_relation" if self.neighbor_dual_history else
                "causal_neighbor" if self.neighbor_causal else "neighbor"
            )
            index_tensors = self.dataset.index(index_name)
            if not isinstance(index_tensors, tuple) or any(
                not isinstance(value, torch.Tensor) for value in index_tensors
            ):
                raise TypeError(f"dataset.index('{index_name}') must return a tensor tuple")
            self._neighbor_index_cpu = index_tensors

        # Each immutable index tensor is copied once per device.
        index_tensors_dev = tuple(
            value if value.device == dev else value.to(dev, non_blocking=True)
            for value in self._neighbor_index_cpu
        )
        self._neighbor_index_cache[key] = index_tensors_dev
        return index_tensors_dev



    def _get_encoder_output(self, e_emb, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids, output_repr=False, return_ctx: bool = False, history_relation_ids: Tensor = None):
        n = p_emb.size(0)
        device = p_emb.device

        # The causal index retains all train events; query-time retrieval must
        # happen before the context-size cap. The legacy branch is retained to
        # re-evaluate saved checkpoints under their original protocol.
        neighbor_index = self._get_neighbor_index(device)
        ctx_list, ctx_size = neighbor_index[:2]
        ids_index = ids.detach()
        if ids_index.dtype != torch.long:
            ids_index = ids_index.long()
        if ids_index.device != ctx_list.device:
            ids_index = ids_index.to(ctx_list.device, non_blocking=True)
        if self.neighbor_causal:
            if (self.get_option("neighbor.sampling") != "time_ordered" or
                    self.get_option("neighbor.order") not in ("time_desc", "desc")):
                raise ValueError("Causal neighbors require time_ordered/time_desc")
            query_times = t_ids.to(ctx_list.device) if isinstance(t_ids, torch.Tensor) else torch.as_tensor(t_ids, device=ctx_list.device)
            if self.neighbor_dual_history:
                if history_relation_ids is None:
                    raise ValueError("Dual history retrieval requires encoded history relation IDs")
                query_history_relations = history_relation_ids.to(ctx_list.device)
                ctx_ids, ctx_size, _ = select_dual_past_context(
                    *neighbor_index, ids_index, query_times,
                    query_history_relations, self.dataset.num_relations(),
                )
            elif self.neighbor_time_stratified_history:
                ctx_ids, ctx_size, _ = select_time_stratified_past_context(
                    ctx_list, ctx_size, ids_index, query_times,
                )
            else:
                ctx_ids, ctx_size, _ = select_past_context(
                    ctx_list, ctx_size, ids_index, query_times, self.max_context_size
                )
        else:
            ctx_ids = ctx_list.index_select(0, ids_index).transpose(1, 2)
            ctx_size = ctx_size.index_select(0, ids_index)
            neighbor_sampling = self.get_option("neighbor.sampling")
            neighbor_order = self.get_option("neighbor.order")
            if self.training and neighbor_sampling == "random":
                perm_vector = sc.get_randperm_from_lengths(ctx_size, ctx_ids.size(1))
                ctx_ids = torch.gather(ctx_ids, 1, perm_vector.unsqueeze(-1).expand_as(ctx_ids))
            else:
                if neighbor_sampling not in ("random", "time_ordered"):
                    raise ValueError(f"Unsupported neighbor.sampling: {neighbor_sampling}")
                if neighbor_sampling == "time_ordered":
                    length = ctx_ids.size(1)
                    ar = torch.arange(length, device=device).unsqueeze(0).expand(n, length)
                    valid = ar < ctx_size.unsqueeze(1)
                    time_key = ctx_ids[..., 2]
                    if neighbor_order in ("time_desc", "desc"):
                        key = torch.where(valid, time_key, time_key.new_full((n, length), -1))
                        order = torch.argsort(key, dim=1, descending=True)
                    elif neighbor_order in ("time_asc", "asc"):
                        pad_val = int(self.dataset.num_times()) + 1
                        key = torch.where(valid, time_key, time_key.new_full((n, length), pad_val))
                        order = torch.argsort(key, dim=1, descending=False)
                    else:
                        raise ValueError(f"Unsupported neighbor.order: {neighbor_order}")
                    ctx_ids = torch.gather(ctx_ids, 1, order.unsqueeze(-1).expand_as(ctx_ids))
            ctx_ids = ctx_ids[:, :self.max_context_size]
            ctx_size = ctx_size.clamp(max=self.max_context_size)

        # [bs, max_ctx_size]
        entity_ids = ctx_ids[...,0]
        relation_ids = ctx_ids[...,1]
        time_ids = ctx_ids[...,2]

        # ------------------------------------------------------------
        # Relative Δt Embedding: represent "how long ago" each neighbor happened
        # Δt = (query_time_id - neighbor_time_id), bucketized with (linear/log) bins
        # We add it to the *time token* of each context triple.
        # ------------------------------------------------------------
        rel_dt_emb = None
        if getattr(self, "rel_dt_enable", False) and self.rel_dt_embed is not None:
            if isinstance(t_ids, torch.Tensor):
                t_ids_dev = t_ids if t_ids.device == device else t_ids.to(device, non_blocking=True)
            else:
                t_ids_dev = torch.as_tensor(t_ids, device=device)

            dt_ids = (t_ids_dev.view(n, 1) - time_ids).clamp(min=0)

            # Zero-out padding positions (so masked slots do not carry spurious large Δt)
            L = time_ids.size(1)
            ar = torch.arange(L, device=device).unsqueeze(0).expand(n, L)
            valid_ctx = ar < ctx_size.unsqueeze(1)  # ctx_size is still "number of kept neighbors"
            dt_ids = torch.where(valid_ctx, dt_ids, dt_ids.new_zeros((1, 1)).expand_as(dt_ids))

            dt_bucket = self._bucketize_rel_dt(dt_ids)
            rel_dt_emb = self.rel_dt_dropout(self.rel_dt_embed(dt_bucket))


        # initialize mask by length of context, seq=[CLS, S, N1, N2, ...]
        # 1 is the positions that will be attended to
        ctx_size = ctx_size + 2
        attention_mask = sc.get_mask_from_sequence_lengths(ctx_size, self.max_context_size + 2)

        if self.training and not output_repr:
            # mask out ground truth during training to avoid overfitting
            gt_mask = ((entity_ids != gt_ent.view(n, 1)) | (relation_ids != gt_rel.view(n, 1))) | (time_ids != gt_tim.view(n, 1))
            ctx_random_mask = (attention_mask
                               .new_ones((n, self.max_context_size))
                               .bernoulli_(1 - self.get_option("ctx_dropout")))
            attention_mask[:,2:] = attention_mask[:, 2:] & ctx_random_mask & gt_mask

        # cache neighbor keep-mask for History-copy bias (Stage-3)
        ctx_keep_mask = attention_mask[:, 2:] if return_ctx else None

        # [bs, max_ctx_size, dim]
        entity_emb = self._entity_embedder().embed(entity_ids)
        relation_emb = self._relation_embedder().embed(relation_ids)
        time_emb = self._time_embedder().embed(time_ids)
        if rel_dt_emb is not None:
            time_emb = time_emb + (self.rel_dt_scale * rel_dt_emb)

        # Apply the timestamp-conditioned block transform to context entity embeddings
        evolver_reg = entity_emb.new_zeros(())
        if self.time_evolver is not None:
            entity_emb, evolver_reg = self.time_evolver(entity_emb, time_emb)

        if self.training and self.get_option("self_dropout") > 0 and self.max_context_size > 0 and not output_repr and self.get_option("add_mlm_loss"):
            # sample a proportion of input for masked prediction similar to the MLM in BERT
            self_dropout_sample = sc.get_bernoulli_mask([n], self.get_option("self_dropout"), device)

            # replace with mask tokens
            masked_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_mask"), device) & self_dropout_sample
            t_emb[masked_sample] = self.local_mask.unsqueeze(0)

            # replace with random sampled entities, no back propagation here
            replaced_sample = sc.get_bernoulli_mask([n], self.get_option(
                "mlm_replace"), device) & self_dropout_sample & ~masked_sample
            t_emb[replaced_sample] = self._time_embedder().embed(torch.randint(self.dataset.num_times(
            ), replaced_sample.shape, dtype=torch.long, device=device))[replaced_sample].detach()

        src = torch.cat([torch.stack([e_emb, p_emb, t_emb], dim=1), torch.stack([entity_emb, relation_emb, time_emb], dim=2)
                         .view(n, 3 * self.max_context_size, self.dim)], dim=1)
        src = src.reshape(n, self.max_context_size + 1, 3, self.dim)

        # only keep un-masked positions to reduce computational cost
        src = src[attention_mask[:, 1:]]

        # add CLS (local) and pos embedding
        pos = self.atomic_type_embeds(torch.arange(0, 4, device=device)).unsqueeze(0).repeat(src.shape[0], 1, 1)
        src = torch.cat([self.cls.expand(src.size(0), 1, self.dim), src], dim=1) + pos

        src = F.dropout(src, p=self.get_option("output_dropout"), training=self.training and not output_repr)
        src = self.atomic_layer_norm(src)

        # [bs, dim]
        out = self.atom_encoder(src,
                                self.convert_mask(src.new_ones(src.size(0), src.size(1), dtype=torch.long)),
                                output_all_encoded_layers=False)[-1][:,0]

        # recover results from output based on mask
        src = out.new_zeros(n, self.max_context_size + 1, self.dim)
        src[attention_mask[:, 1:]] = out

        # when not using graph context, exit here
        if self.max_context_size == 0:
            if return_ctx:
                empty_attention = src.new_zeros((n, 0))
                return src[:, 0], 0, evolver_reg, entity_ids, relation_ids, time_ids, ctx_keep_mask, empty_attention
            else:
                return src[:, 0], 0, evolver_reg
        
        # out1 = src

        # # begin the processing of graph context with the upper transformer block
        # # add CLS (global) and pos embeddings
        # src = torch.cat([self.global_cls.expand(n, 1, self.dim), src], dim=1)

        # # MLP-Mixer
        # if head_or_tail == True:
        #     s_p = torch.ones_like(src, requires_grad=True)
        # else:
        #     s_p = torch.zeros_like(src, requires_grad=True)

        # cls = self.global_cls.expand(n, self.max_context_size + 1, self.dim)

        # src = torch.stack([cls, src, s_p], dim=1)
        src = F.dropout(src, p=self.get_option("hidden_dropout"), training=self.training)
        src = self.layer_norm(src)

        out = self.mixer_encoder(src)

        # Stage-2 pooling: RCAP (relation-guided attention) or mask-aware mean
        event_mask = attention_mask[:, 1:]  # [bs, max_ctx+1], corresponds to (S + neighbors)
        history_attention = None

        if getattr(self, "dra_enable", False) and self.dra_q is not None:
            dk = float(self.dra_key_dim)
            q = self.dra_q(p_emb)                  # [bs, dk]
            k = self.dra_k(out)                    # [bs, L, dk]
            logits = torch.einsum("bd,bld->bl", q, k) / math.sqrt(dk)
            logits = logits.masked_fill(~event_mask, -1e9)
            raw_attn = torch.softmax(logits, dim=1)
            attn = raw_attn
            if self.dra_attn_dropout is not None:
                attn = self.dra_attn_dropout(attn)
            out_pooling = torch.einsum("bl,bld->bd", attn, out)  # [bs, dim]
            if return_ctx:
                history_logits = logits[:, 1:].masked_fill(~ctx_keep_mask, -1e9)
                history_attention = torch.softmax(history_logits, dim=1)
                history_attention = history_attention * ctx_keep_mask.to(history_attention.dtype)
        else:
            denom = event_mask.sum(dim=1, keepdim=True).clamp(min=1)
            out_pooling = (out * event_mask.unsqueeze(-1)).sum(dim=1) / denom
            if return_ctx:
                history_attention = ctx_keep_mask.to(out.dtype)
        if return_ctx:
            history_attention = history_attention / history_attention.sum(dim=1, keepdim=True).clamp(min=1e-12)
        # return out[:, 0], 0

        # out = out + out1


        # # begin the processing of graph context with the upper transformer block
        # # add CLS (global) and pos embeddings
        # src = torch.cat([self.global_cls.expand(n, 1, self.dim), src], dim=1)
        # pos = self.type_embeds(torch.arange(0, 3, device=device))
        # src[:, 0] = src[:, 0] + pos[0].unsqueeze(0)
        # src[:, 1] = src[:, 1] + pos[1].unsqueeze(0)
        # src[:, 2:] = src[:, 2:] + pos[2].view(1, 1, -1)

        # src = F.dropout(src, p=self.get_option("hidden_dropout"), training=self.training)
        # src = self.layer_norm(src)
        # out = self.transformer_encoder(src, None, self.convert_mask_rat(attention_mask))

        # if output_repr:
        #     return out, self.convert_mask(attention_mask)

        # out = out[-1][:,:2]

        # compute the mlm-like loss if needed
        if self.training and self.get_option("add_mlm_loss") and self.get_option("self_dropout") > 0.0 and self_dropout_sample.sum() > 0:
            all_time_emb = self._time_embedder().embed_all()
            all_time_emb = F.dropout(all_time_emb, p=self.get_option("output_dropout"), training=self.training)
            # source_scores = self.similarity(out[:, 3], all_time_emb, False).view(n, -1)
            source_scores = self.similarity(out_pooling, all_time_emb, False).view(n, -1)
            self_pred_loss = F.cross_entropy(
                source_scores[self_dropout_sample], t_ids[self_dropout_sample], reduction='mean')
            # return out[:, 0], self_pred_loss
            if return_ctx:
                return out_pooling, 2*self_pred_loss, evolver_reg, entity_ids, relation_ids, time_ids, ctx_keep_mask, history_attention
            return out_pooling, 2*self_pred_loss, evolver_reg
        else:
            # return out[:, 0], 0
            if return_ctx:
                return out_pooling, 0, evolver_reg, entity_ids, relation_ids, time_ids, ctx_keep_mask, history_attention
            else:
                return out_pooling, 0, evolver_reg

    def convert_mask_rat(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def convert_mask(self, attention_mask):
        # extend mask to Transformer format
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask


    def _history_gate_value(
        self, query_relation: Tensor, local_repr: Tensor, query_time_ids: Tensor,
        ctx_time_ids: Tensor, ctx_keep_mask: Tensor, ctx_attention: Tensor,
    ) -> Tensor:
        """Predict a query-specific HEPE multiplier using available history only."""
        if self.history_gate is None:
            return local_repr.new_ones((local_repr.size(0), 1))
        mask = ctx_keep_mask.to(local_repr.dtype)
        count = mask.sum(dim=1, keepdim=True)
        if self.history_gate_time_feature == "past_age":
            age = (query_time_ids.unsqueeze(1) - ctx_time_ids).clamp(min=0).to(local_repr.dtype)
            mean_age = (age * mask).sum(dim=1, keepdim=True) / count.clamp(min=1)
        else:
            # The uploaded source can select later train events, and its
            # ICEWS05 time IDs are first-occurrence IDs rather than dates.
            # Count, attention entropy and query representations remain valid.
            mean_age = local_repr.new_zeros((local_repr.size(0), 1))
        alpha = ctx_attention * mask
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-12)
        entropy = -(alpha * alpha.clamp(min=1e-12).log()).sum(dim=1, keepdim=True)
        features = torch.cat(
            (query_relation, local_repr, count.log1p(), mean_age.log1p(), entropy), dim=1
        )
        return 2.0 * torch.sigmoid(self.history_gate(features))

    def _apply_history_copy_bias(
        self,
        target_scores: Tensor,
        query_time_ids: Tensor,
        ctx_entity_ids: Tensor,
        ctx_time_ids: Tensor,
        ctx_keep_mask: Tensor,
        ctx_attention: Tensor = None,
        query_gate: Tensor = None,
        candidate_ids: Tensor = None,
    ) -> Tensor:
        """Add a History-copy logit bias to target_scores.

        This is a lightweight Copy–Generation hybrid:
        - 'Generation' is the base score from GCFF/RCAP.
        - 'Copy' adds a bias for entities in the selected train-only neighbor list.

        The bias is computed purely from discrete IDs (no extra forward passes), and can
        be time-decayed by (time_gamma ** Δt).

        Args:
            target_scores: [bs, C] (C may be a chunk via entity_subset).
            query_time_ids: [bs]
            ctx_entity_ids: [bs, L]
            ctx_time_ids: [bs, L]
            ctx_keep_mask: [bs, L] (bool; true for kept neighbors)
            candidate_ids: Optional [C] tensor that maps columns to entity IDs (entity_subset).
        """
        if (not getattr(self, "history_copy_enable", False)) or self.history_copy_weight == 0.0:
            return target_scores
        if ctx_entity_ids is None or ctx_keep_mask is None or query_time_ids is None:
            return target_scores
        if ctx_entity_ids.numel() == 0 or target_scores.numel() == 0:
            return target_scores

        bs, C = target_scores.shape
        device = target_scores.device
        dtype = target_scores.dtype

        # Ensure device alignment
        if query_time_ids.device != device:
            query_time_ids = query_time_ids.to(device, non_blocking=True)
        if ctx_entity_ids.device != device:
            ctx_entity_ids = ctx_entity_ids.to(device, non_blocking=True)
        if ctx_time_ids.device != device:
            ctx_time_ids = ctx_time_ids.to(device, non_blocking=True)
        if ctx_keep_mask.device != device:
            ctx_keep_mask = ctx_keep_mask.to(device, non_blocking=True)
        if ctx_attention is not None and ctx_attention.device != device:
            ctx_attention = ctx_attention.to(device, non_blocking=True)
        if query_gate is not None and query_gate.device != device:
            query_gate = query_gate.to(device, non_blocking=True)
        if candidate_ids is not None and candidate_ids.device != device:
            candidate_ids = candidate_ids.to(device, non_blocking=True)

        mask = ctx_keep_mask
        if getattr(self, "neighbor_causal", False):
            mask = mask & (ctx_time_ids < query_time_ids.view(bs, 1))
        w = mask.to(dtype=torch.float32)

        # Optional time decay: w *= gamma ** Δt
        gamma = float(getattr(self, "history_copy_time_gamma", 1.0))
        if gamma < 1.0:
            # Δt = query_t - neighbor_t (should already be non-negative under HistoryMask)
            dt = (query_time_ids.view(bs, 1) - ctx_time_ids).clamp(min=0).to(torch.float32)
            w = w * torch.exp(dt * math.log(gamma))

        if self.history_copy_mode == "adaptive":
            if ctx_attention is None:
                raise ValueError("Adaptive HEPE requires historical attention weights")
            alpha = ctx_attention.to(torch.float32) * mask.to(torch.float32)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-12)
            count = mask.sum(dim=1, keepdim=True).to(torch.float32)
            relative_weight = (1.0 - self.history_attention_mix) + self.history_attention_mix * count * alpha
            w = w * relative_weight

        # Normalization (stabilizes magnitude across variable neighbor counts)
        norm = str(getattr(self, "history_copy_normalize", "mean")).lower()
        if norm == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
            w = w / denom
        elif norm == "sum":
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            w = w / denom
        elif norm in ("none", "off"):
            pass
        else:
            raise ValueError(f"rift.hepe.normalize must be one of 'mean'/'sum'/'none', got '{norm}'")

        w = w.to(dtype) * float(self.history_copy_weight)
        if self.history_copy_mode == "adaptive" and query_gate is not None:
            w = w * query_gate.to(dtype)

        # Pairwise case (C==1) – only apply if caller supplies candidate_ids=[bs]
        if C == 1 and candidate_ids is not None and candidate_ids.ndim == 1 and candidate_ids.numel() == bs:
            cand = candidate_ids.view(bs, 1)
            match = (ctx_entity_ids == cand) & mask
            bias = (w * match.to(dtype)).sum(dim=1, keepdim=True)
            return target_scores + bias

        # Chunk / full ranking case
        if candidate_ids is None:
            # Columns correspond to global entity IDs 0..num_entities-1
            idx = ctx_entity_ids.long()
            src = w
            # Guard against any stray padding values (masked entries have src=0 anyway)
            idx = idx.clamp(min=0, max=C - 1)
            return target_scores.scatter_add(1, idx, src)

        # Map ctx_entity_ids -> column indices in candidate_ids
        cand = candidate_ids.long().view(-1)
        if cand.numel() != C:
            raise ValueError("candidate_ids must have length equal to target_scores.size(1)")

        # Fast path: contiguous range
        start = int(cand[0].item())
        if int(cand[-1].item()) - start == C - 1:
            ar = torch.arange(C, device=device, dtype=torch.long) + start
            if torch.equal(cand, ar):
                col = ctx_entity_ids.long() - start
                in_range = (col >= 0) & (col < C) & mask
                idx = torch.where(in_range, col, col.new_zeros(1, 1).expand_as(col))
                src = w * in_range.to(dtype)
                return target_scores.scatter_add(1, idx, src)

        # General mapping (handles arbitrary subsets)
        # Build a lookup table of size num_entities (ICEWS is small; this is practical).
        num_ent = self.dataset.num_entities() if callable(getattr(self.dataset, "num_entities", None)) else int(self.dataset.num_entities)
        lut = torch.full((num_ent,), -1, device=device, dtype=torch.long)
        lut[cand.clamp(min=0, max=num_ent - 1)] = torch.arange(C, device=device, dtype=torch.long)
        col = lut[ctx_entity_ids.long().clamp(min=0, max=num_ent - 1)]
        in_range = (col >= 0) & mask
        idx = torch.where(in_range, col, col.new_zeros(1, 1).expand_as(col))
        src = w * in_range.to(dtype)
        return target_scores.scatter_add(1, idx, src)

    def _scoring(self, s_emb, p_emb, o_emb, t_emb, is_pairwise, ids, gt_ent, gt_rel, gt_tim, t_ids, candidate_ids=None, history_relation_ids=None):
        # Local context encoding (S + ordered neighbors)
        use_copy = bool(getattr(self, "history_copy_enable", False)) and (
            (not self.training) or bool(getattr(self, "history_copy_apply_in_training", False))
        )
        if use_copy:
            local_repr, self_pred_loss, evolver_reg, ctx_ent, ctx_rel, ctx_tim, ctx_keep, ctx_attention = self._get_encoder_output(
                s_emb, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids,
                return_ctx=True, history_relation_ids=history_relation_ids,
            )
        else:
            local_repr, self_pred_loss, evolver_reg = self._get_encoder_output(
                s_emb, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids,
                history_relation_ids=history_relation_ids,
            )
            ctx_ent = ctx_rel = ctx_tim = ctx_keep = ctx_attention = None


        # GCFF fusion (global entity embedding + local context + time [+ relation])
        fused_repr = self._fuse_gltf(local_repr, s_emb, p_emb, t_emb)

        # Score against candidate objects
        o_emb = F.dropout(o_emb, p=self.get_option("output_dropout"), training=self.training)
        target_scores = self.similarity(fused_repr, o_emb, is_pairwise).view(p_emb.size(0), -1)

        # Stage-3: HEPE bias (copy historical entities into the score space)
        if use_copy and isinstance(t_ids, torch.Tensor):
            query_gate = None
            if self.history_copy_mode == "adaptive":
                query_gate = self._history_gate_value(
                    p_emb, local_repr, t_ids, ctx_tim, ctx_keep, ctx_attention
                )
            target_scores = self._apply_history_copy_bias(
                target_scores,
                t_ids,
                ctx_ent,
                ctx_tim,
                ctx_keep,
                ctx_attention=ctx_attention,
                query_gate=query_gate,
                candidate_ids=candidate_ids,
            )

        if self.training:
            # CDC (contrastive) on subsampled batch
            cdc_loss = self._maybe_compute_cdc_loss(
                fused_repr, s_emb, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids,
                history_relation_ids=history_relation_ids,
            )
            return target_scores, self_pred_loss, evolver_reg, cdc_loss
        else:
            return target_scores

    def score_emb(self, s_emb, p_emb, o_emb, t_emb, combine: str, s, o, gt_ent=None, gt_rel=None, gt_tim=None, t_ids=None, candidate_ids: Tensor = None, history_relation_ids: Tensor = None):
        # the 'spo' combination is only used in reciprocal model
        # 'spo' prefix stands for triple-wise scoring, suffix "s"/"o" indicates direction
        if combine == 'spoo' or combine == 'sp_' or combine == 'spo':
            out = self._scoring(s_emb, p_emb, o_emb, t_emb, combine.startswith('spo'), s, gt_ent, gt_rel, gt_tim, t_ids, candidate_ids=candidate_ids, history_relation_ids=history_relation_ids)
        elif combine == 'spos' or combine == '_po':
            out = self._scoring(o_emb, p_emb, s_emb, t_emb, combine.startswith('spo'), o, gt_ent, gt_rel, gt_tim, t_ids, candidate_ids=candidate_ids, history_relation_ids=history_relation_ids)
        else:
            raise Exception("Combine {} is not supported in RIFT's score function".format(combine))
        return out


class RIFT(KgeModel):
    r"""Implementation of the RIFT KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, RIFTScorer, configuration_key=configuration_key
        )
        self.loss = KgeLoss.create(config)

    def forward(self, fn_name, *args, **kwargs):
        # bind entity/relation embedder to scorer to retrieve embeddings
        self._scorer._entity_embedder = self.get_s_embedder
        self._scorer._relation_embedder = self.get_p_embedder
        self._scorer._time_embedder = self.get_t_embedder

        # call score_sp/score_po during training, score_spo/score_sp_po during inference
        scores = getattr(self, fn_name)(*args, **kwargs)

        # delete references to embedder getter
        del self._scorer._entity_embedder
        del self._scorer._relation_embedder
        del self._scorer._time_embedder

        if fn_name == 'get_rift_repr':
            return scores

        if self.training:
            self_loss_w = self.get_option("self_dropout")
            # MLM-like loss is weighted by the proportion of entities sampled
            self_loss_w = self_loss_w / (1 + self_loss_w)
            time_evolver_w = self.get_option("cte.stability_reg") if self.get_option("cte.enable") else 0.0
            cdc_w = self.get_option("cdc.weight") if self.get_option("cdc.enable") else 0.0
            bs = scores[0].size(0)
            return (
                self.loss(scores[0], kwargs["gt_ent"])
                + self_loss_w * scores[1] * bs
                + time_evolver_w * scores[2] * bs
                + cdc_w * scores[3] * bs
            )
        else:
            return scores

    def get_rift_repr(self, s, p):
        """Return the encoded representation for (s, p) using a dummy time=0.

        This is primarily used by analysis / rule tooling; it should be safe and not crash.
        """
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        t_ids = torch.zeros_like(s)
        t_emb = self.get_t_embedder().embed(t_ids)
        # output_repr=True disables MLM objective while still producing the representation
        rep, _, _ = self._scorer._get_encoder_output(
            s_emb, p_emb, t_emb,
            s,                       # ids for neighbor index lookup
            s, p, t_ids,             # dummy ground-truth placeholders (ignored when output_repr=True)
            t_ids,
            output_repr=True,
            history_relation_ids=p + self.dataset.num_relations(),
        )
        return rep

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if direction:
            if direction == 's':
                p_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
                candidate_ids = s if self.get_option("hepe.correct_pairwise") else None
            else:
                p_emb = self.get_p_embedder().embed(p)
                candidate_ids = o if self.get_option("hepe.correct_pairwise") else None
            return self._scorer.score_emb(
                s_emb, p_emb, o_emb, t_emb, "spo" + direction,
                s, o, None, None, None, t, candidate_ids=candidate_ids,
                history_relation_ids=p if direction == 's' else p + self.dataset.num_relations(),
            ).view(-1)
        else:
            raise Exception(
                "The RIFT model cannot compute "
                "undirected spo scores."
            )

    def score_sp(self, s: Tensor, p: Tensor, t: Tensor, o: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        t_emb = self.get_t_embedder().embed(t)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(
            s_emb, p_emb, o_emb, t_emb, "sp_", s, None, gt_ent, gt_rel, gt_tim, t,
            history_relation_ids=p + self.dataset.num_relations(),
        )

    def score_po(self, p: Tensor, o: Tensor, t: Tensor, s: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())

        return self._scorer.score_emb(
            s_emb, p_inv_emb, o_emb, t_emb, "_po", None, o, gt_ent, gt_rel, gt_tim, t,
            history_relation_ids=p,
        )

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, entity_subset: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(
                s_emb, p_emb, all_entities, t_emb, "sp_", s, None,
                t_ids=t, candidate_ids=entity_subset,
                history_relation_ids=p + self.dataset.num_relations(),
            )
            po_scores = self._scorer.score_emb(
                all_entities, p_inv_emb, o_emb, t_emb, "_po", None, o,
                t_ids=t, candidate_ids=entity_subset, history_relation_ids=p,
            )
        else:
            assert False
        return torch.cat((sp_scores, po_scores), dim=1)


