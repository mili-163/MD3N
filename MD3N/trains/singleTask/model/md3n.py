import functools
import math
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal

from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .rcan import Group
from .scoremodel import Euler_Maruyama_sampler, ScoreNet, loss_fn

__all__ = ["MD3N", "IMDER"]


class ResidualAdapter(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class CrossModalInjector(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query_state, semantic_state, scale=1.0):
        if scale <= 0:
            return query_state
        query = query_state.permute(2, 0, 1)
        semantic_token = semantic_state.unsqueeze(0)
        attended, _ = self.attn(query, semantic_token, semantic_token)
        injected = self.mlp(attended).permute(1, 2, 0)
        return query_state + scale * injected


class ResidualRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            Group(num_channels=channels * 2, num_blocks=8, reduction=8),
            nn.Conv1d(channels * 2, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class PrototypeMixturePrior(nn.Module):
    def __init__(self, proto_dim, num_classes, num_prototypes, low_rank_rank):
        super().__init__()
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.low_rank_rank = low_rank_rank

        self.prototype_logits = nn.Parameter(torch.zeros(num_classes, num_prototypes))
        self.means = nn.Parameter(torch.randn(num_classes, num_prototypes, proto_dim) * 0.02)
        self.cov_factor = nn.Parameter(
            torch.randn(num_classes, num_prototypes, proto_dim, low_rank_rank) * 0.01
        )
        self.log_cov_diag = nn.Parameter(torch.zeros(num_classes, num_prototypes, proto_dim))

    def _distribution(self):
        cov_diag = F.softplus(self.log_cov_diag) + 1e-4
        return LowRankMultivariateNormal(self.means, self.cov_factor, cov_diag)

    def log_prob(self, z):
        log_weights = F.log_softmax(self.prototype_logits.reshape(-1), dim=0).view(
            self.num_classes, self.num_prototypes
        )
        component_log_prob = self._distribution().log_prob(z[:, None, None, :])
        return torch.logsumexp(component_log_prob + log_weights, dim=(1, 2))

    def responsibilities(self, z):
        log_weights = F.log_softmax(self.prototype_logits.reshape(-1), dim=0).view(
            self.num_classes, self.num_prototypes
        )
        component_log_prob = self._distribution().log_prob(z[:, None, None, :])
        logits = component_log_prob + log_weights
        post = torch.softmax(logits.reshape(z.size(0), -1), dim=-1)
        return post.view(z.size(0), self.num_classes, self.num_prototypes)

    def separation_loss(self, tau):
        means = self.means.reshape(-1, self.proto_dim)
        if means.size(0) < 2:
            return means.new_zeros(())
        pairwise_sq = torch.cdist(means, means, p=2).pow(2)
        mask = ~torch.eye(pairwise_sq.size(0), dtype=torch.bool, device=pairwise_sq.device)
        return torch.exp(-pairwise_sq[mask] / tau).mean()


def marginal_prob_std(t, sigma):
    t = torch.as_tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    return torch.as_tensor(sigma ** t, device=t.device if isinstance(t, torch.Tensor) else None)


def _scheduled_by_mr(args, key, default):
    schedule = args.get(f"{key}_by_mr", None)
    if not schedule or "mr" not in args:
        return default
    mr = float(args.get("mr", 0.0))
    for boundary, value in schedule:
        if mr <= float(boundary) + 1e-8:
            return value
    return schedule[-1][1]


def _normalize_stage_boundary(value, diffusion_steps):
    value = float(value)
    if value > 1.0:
        return value / float(diffusion_steps)
    return value


class MD3N(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )
        self.use_bert = args.use_bert

        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        self.diffusion_steps = int(args.get("diffusion_steps", 1000))
        self.stage_t1 = _normalize_stage_boundary(args.get("stage_t1", 200), self.diffusion_steps)
        self.stage_t2 = _normalize_stage_boundary(args.get("stage_t2", 800), self.diffusion_steps)
        self.guidance_scale = float(args.get("guidance_scale", 0.35))
        self.lambda_sep = float(args.get("lambda_sep", 0.05))
        self.lambda_rcp = float(args.get("lambda_rcp", 0.1))
        self.lambda_int_max = float(args.get("lambda_int_max", 0.35))
        self.semantic_step_size = float(args.get("semantic_step_size", 0.2))
        self.proto_tau = float(args.get("proto_tau", 1.0))
        self.sample_steps = int(args.get("sample_steps", 1000))
        self.init_noise_scale = float(
            _scheduled_by_mr(args, "init_noise_scale", args.get("init_noise_scale", 1.0))
        )
        self.stage_weights = args.get("stage_weights", [1.0, 1.0, 1.0])

        text_len, audio_len, vision_len = args.get("seq_lens", (50, 50, 50))
        self.seq_l = max(1, text_len - args.conv1d_kernel_size_l + 1)
        self.seq_a = max(1, audio_len - args.conv1d_kernel_size_a + 1)
        self.seq_v = max(1, vision_len - args.conv1d_kernel_size_v + 1)

        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        self.score_l = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_v = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_a = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)

        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False
        )

        adapter_hidden = int(args.get("adapter_hidden", self.d_l * 2))
        self.observed_adapters = nn.ModuleDict(
            {name: ResidualAdapter(self.d_l, adapter_hidden) for name in ["l", "a", "v"]}
        )
        self.target_adapters = nn.ModuleDict(
            {name: ResidualAdapter(self.d_l, adapter_hidden) for name in ["l", "a", "v"]}
        )
        self.missing_adapters = nn.ModuleDict(
            {name: ResidualAdapter(self.d_l, adapter_hidden) for name in ["l", "a", "v"]}
        )
        self.injectors = nn.ModuleDict(
            {name: CrossModalInjector(self.d_l, num_heads=4) for name in ["l", "a", "v"]}
        )
        self.decoders = nn.ModuleDict(
            {
                "l": nn.Conv1d(self.d_l, self.orig_d_l, kernel_size=1),
                "a": nn.Conv1d(self.d_a, self.orig_d_a, kernel_size=1),
                "v": nn.Conv1d(self.d_v, self.orig_d_v, kernel_size=1),
            }
        )
        self.reciprocal_net = nn.Sequential(
            nn.Linear(self.d_l, self.d_l * 2),
            nn.GELU(),
            nn.Linear(self.d_l * 2, self.d_l),
        )

        self.missing_tokens = nn.ParameterDict(
            {
                "l": nn.Parameter(torch.zeros(1, self.d_l, self.seq_l)),
                "a": nn.Parameter(torch.zeros(1, self.d_a, self.seq_a)),
                "v": nn.Parameter(torch.zeros(1, self.d_v, self.seq_v)),
            }
        )

        proto_dim = int(args.get("proto_dim", 64))
        proto_rank = int(args.get("proto_rank", 4))
        num_proto_per_class = int(args.get("num_proto_per_class", 2))
        self.prototype_projector = nn.Sequential(
            nn.Linear(self.d_l, self.d_l * 2),
            nn.GELU(),
            nn.Linear(self.d_l * 2, proto_dim),
        )
        self.semantic_to_condition = nn.Linear(proto_dim, self.d_l)
        nn.init.zeros_(self.prototype_projector[-1].weight)
        nn.init.zeros_(self.prototype_projector[-1].bias)
        nn.init.zeros_(self.semantic_to_condition.weight)
        nn.init.zeros_(self.semantic_to_condition.bias)
        self.prototype_prior = PrototypeMixturePrior(
            proto_dim=proto_dim,
            num_classes=int(args.get("num_classes", 3)),
            num_prototypes=num_proto_per_class,
            low_rank_rank=proto_rank,
        )

        self.refiners = nn.ModuleDict(
            {
                "l": self._make_refiner(self.d_l),
                "a": self._make_refiner(self.d_a),
                "v": self._make_refiner(self.d_v),
            }
        )

        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")
        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")
        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def _make_refiner(self, channels):
        return ResidualRefiner(channels)

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def _align_raw_target(self, raw_state, projected_state, kernel_size):
        aligned_len = projected_state.size(-1)
        start = min(max(0, kernel_size // 2), max(0, raw_state.size(-1) - aligned_len))
        end = start + aligned_len
        return raw_state[:, :, start:end]

    def _encode_modalities(self, text, audio, video):
        if self.use_bert:
            if self.training and self.text_model.use_finetune:
                text = self.text_model(text)
            else:
                with torch.no_grad():
                    text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_l = self.proj_l(x_l)
        proj_a = self.proj_a(x_a)
        proj_v = self.proj_v(x_v)

        observed_states = {
            "l": self.observed_adapters["l"](proj_l),
            "a": self.observed_adapters["a"](proj_a),
            "v": self.observed_adapters["v"](proj_v),
        }
        target_states = {
            "l": self.target_adapters["l"](proj_l),
            "a": self.target_adapters["a"](proj_a),
            "v": self.target_adapters["v"](proj_v),
        }
        raw_targets = {
            "l": self._align_raw_target(x_l, proj_l, self.proj_l.kernel_size[0]),
            "a": self._align_raw_target(x_a, proj_a, self.proj_a.kernel_size[0]),
            "v": self._align_raw_target(x_v, proj_v, self.proj_v.kernel_size[0]),
        }
        return observed_states, target_states, raw_targets

    def _stage_from_time(self, time_value):
        if time_value > self.stage_t2:
            return 1
        if time_value > self.stage_t1:
            return 2
        return 3

    def _semantic_injection_scale(self, time_value):
        if time_value > self.stage_t2:
            denom = max(1e-6, 1.0 - self.stage_t2)
            return self.lambda_int_max * max(0.0, 1.0 - (time_value - self.stage_t2) / denom)
        return self.lambda_int_max

    def _guidance_schedule(self, time_value):
        if time_value > self.stage_t2:
            return 0.0
        if time_value > self.stage_t1:
            return self.guidance_scale
        t1 = max(self.stage_t1, 1e-6)
        anneal = 0.5 * (1.0 + math.cos(math.pi * (1.0 - time_value / t1)))
        return self.guidance_scale * anneal

    def _pool_state(self, state):
        return state.mean(dim=-1)

    def _missing_state(self, modality_name, batch_size, available_context=None):
        token = self.missing_tokens[modality_name].expand(batch_size, -1, -1)
        if available_context is not None:
            token = token + available_context[..., : token.size(-1)]
        return self.missing_adapters[modality_name](token)

    def _build_condition(self, available_context, proto_z, seq_len, time_value):
        semantic_seq = self.semantic_to_condition(proto_z).unsqueeze(-1).expand(-1, -1, seq_len)
        return available_context + self._semantic_injection_scale(time_value) * semantic_seq

    def _bidirectional_update(self, proto_z, latent_state):
        tracked_proto = self.prototype_projector(self._pool_state(latent_state).detach())
        cosine = F.cosine_similarity(proto_z, tracked_proto, dim=-1, eps=1e-6).unsqueeze(-1)
        updated = proto_z + self.semantic_step_size * (tracked_proto - cosine * proto_z)
        return F.normalize(updated, dim=-1)

    def _prototype_guidance(self, latent_state):
        pooled = self._pool_state(latent_state)
        proto_z = self.prototype_projector(pooled)
        log_density = self.prototype_prior.log_prob(proto_z).sum()
        return torch.autograd.grad(log_density, latent_state, retain_graph=False, create_graph=False)[0]

    def _sample_modalities(self, num_modal):
        modal_idx = [0, 1, 2]  # 0:text, 1:vision, 2:audio
        available_modalities = sample(modal_idx, num_modal)
        missing_modalities = [idx for idx in modal_idx if idx not in available_modalities]
        return available_modalities, missing_modalities

    def _fuse_modalities(self, latent_states):
        proj_x_l = latent_states["l"].permute(2, 0, 1)
        proj_x_a = latent_states["a"].permute(2, 0, 1)
        proj_x_v = latent_states["v"].permute(2, 0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]

        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        fused = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        fused = self.proj2(
            F.dropout(F.relu(self.proj1(fused), inplace=True), p=self.output_dropout, training=self.training)
        ) + fused
        output = self.out_layer(fused)
        return output, last_h_l, last_h_a, last_h_v, fused

    def forward(self, text, audio, video, num_modal=None):
        batch_size = text.size(0)
        device = text.device
        zero = torch.zeros((), device=device)
        time_value = float(torch.rand(1, device=device).item()) if self.training else 0.5
        stage_id = self._stage_from_time(time_value)

        observed_states, target_states, raw_targets = self._encode_modalities(text, audio, video)
        if num_modal is None:
            num_modal = 3
        available_modalities, missing_modalities = self._sample_modalities(num_modal)
        modality_index_to_name = {0: "l", 1: "v", 2: "a"}

        observed_map = {}
        missing_map = {}
        pooled_observed = []
        sequence_observed = []
        for idx, modality_name in modality_index_to_name.items():
            if idx in available_modalities:
                observed_map[modality_name] = observed_states[modality_name]
                pooled_observed.append(self._pool_state(observed_states[modality_name]))
                sequence_observed.append(observed_states[modality_name])
            else:
                missing_map[modality_name] = None

        observed_reference = torch.stack(pooled_observed, dim=0).mean(dim=0)
        available_context = torch.stack(sequence_observed, dim=0).mean(dim=0)
        for modality_name in missing_map:
            missing_map[modality_name] = self._missing_state(
                modality_name, batch_size, available_context=available_context
            )

        proto_z = self.prototype_projector(observed_reference)
        align_loss = -self.prototype_prior.log_prob(proto_z).mean()
        sep_loss = self.prototype_prior.separation_loss(self.proto_tau)
        if stage_id == 1:
            stage_loss = align_loss
        elif stage_id == 2:
            stage_loss = align_loss + self.lambda_sep * sep_loss
        else:
            stage_loss = self._guidance_schedule(time_value) * align_loss

        recovered_map = {}
        score_loss = zero
        rec_loss = zero
        rcp_loss = zero
        for modality_name in missing_map:
            injected_missing = self.injectors[modality_name](
                missing_map[modality_name],
                self.semantic_to_condition(proto_z),
                self._semantic_injection_scale(time_value),
            )
            if stage_id == 2:
                proto_z = self._bidirectional_update(proto_z, injected_missing)
            condition = self._build_condition(available_context, proto_z, injected_missing.size(-1), time_value)
            score_model = getattr(self, f"score_{modality_name}")
            score_loss = score_loss + loss_fn(
                score_model, target_states[modality_name], self.marginal_prob_std_fn, condition=condition
            )
            recovered = Euler_Maruyama_sampler(
                score_model,
                self.marginal_prob_std_fn,
                self.diffusion_coeff_fn,
                batch_size=batch_size,
                num_steps=self.sample_steps,
                device=device,
                condition=condition,
                init_x=injected_missing,
                init_noise_scale=self.init_noise_scale,
                guidance_fn=self._prototype_guidance,
                guidance_scale=self.guidance_scale,
                guidance_schedule=self._guidance_schedule,
            )
            recovered = self.refiners[modality_name](recovered)
            if stage_id == 2:
                proto_z = self._bidirectional_update(proto_z, recovered)
            recovered_map[modality_name] = recovered

            decoded = self.decoders[modality_name](recovered)
            rec_loss = rec_loss + F.l1_loss(decoded, raw_targets[modality_name])
            rcp_loss = rcp_loss + F.mse_loss(
                self.reciprocal_net(self._pool_state(recovered)), observed_reference.detach()
            )

        if missing_map:
            divisor = float(len(missing_map))
            score_loss = score_loss / divisor
            rec_loss = rec_loss / divisor
            rcp_loss = rcp_loss / divisor
        else:
            score_loss = zero
            rec_loss = zero
            rcp_loss = zero
        end_loss = rec_loss + self.lambda_rcp * rcp_loss

        final_states = {}
        for modality_name in ["l", "a", "v"]:
            if modality_name in observed_map:
                final_states[modality_name] = observed_map[modality_name]
            else:
                final_states[modality_name] = recovered_map[modality_name]

        output, last_h_l, last_h_a, last_h_v, fused = self._fuse_modalities(final_states)
        res = {
            "Feature_t": last_h_l,
            "Feature_a": last_h_a,
            "Feature_v": last_h_v,
            "Feature_f": fused,
            "loss_score": score_loss,
            "loss_stage": stage_loss,
            "loss_end": end_loss,
            "loss_align": align_loss,
            "loss_sep": sep_loss if stage_id == 2 else zero,
            "loss_rec": rec_loss,
            "loss_rcp": rcp_loss,
            "stage_weight": float(self.stage_weights[stage_id - 1]),
            "stage_id": stage_id,
            "ava_modal_idx": available_modalities,
            "M": output,
        }
        return res


IMDER = MD3N
