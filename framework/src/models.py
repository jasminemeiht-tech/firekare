from __future__ import annotations

import torch
from torch import nn


def _pick_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        groups = _pick_group_count(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.squeeze(-1)


class TrialEncoder(nn.Module):
    def __init__(
        self,
        emg_channels: int,
        mech_channels: int,
        tabular_dim: int,
        emg_hidden: int,
        mech_hidden: int,
        tabular_hidden: int,
        fusion_hidden: int,
        action_embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.emg_encoder = ConvEncoder1D(emg_channels, emg_hidden)
        self.mech_encoder = ConvEncoder1D(mech_channels, mech_hidden)
        self.tabular_encoder = None
        tabular_out = 0
        if tabular_dim > 0:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, tabular_hidden),
                nn.LayerNorm(tabular_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            tabular_out = tabular_hidden
        self.action_embedding = nn.Embedding(2, action_embedding_dim)
        self.output_dim = emg_hidden + mech_hidden + action_embedding_dim + tabular_out

    def forward(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emg_latent = self.emg_encoder(emg)
        mech_latent = self.mech_encoder(mechanics)
        action_latent = self.action_embedding(action)
        features = [emg_latent, mech_latent, action_latent]
        if self.tabular_encoder is not None:
            if tabular is None:
                raise ValueError("tabular features are required for the configured hybrid model")
            features.append(self.tabular_encoder(tabular))
        return torch.cat(features, dim=1)


class MultiModalCNN(nn.Module):
    def __init__(
        self,
        emg_channels: int,
        mech_channels: int,
        tabular_dim: int,
        emg_hidden: int,
        mech_hidden: int,
        tabular_hidden: int,
        fusion_hidden: int,
        action_embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.trial_encoder = TrialEncoder(
            emg_channels=emg_channels,
            mech_channels=mech_channels,
            tabular_dim=tabular_dim,
            emg_hidden=emg_hidden,
            mech_hidden=mech_hidden,
            tabular_hidden=tabular_hidden,
            fusion_hidden=fusion_hidden,
            action_embedding_dim=action_embedding_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.trial_encoder.output_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
        biomarker: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self.trial_encoder(emg, mechanics, action, tabular)
        return self.classifier(fused).squeeze(1)


class BiomarkerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        dims = [int(input_dim), *[int(x) for x in hidden_dims if int(x) > 0]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        last_dim = dims[-1]
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(last_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        return self.classifier(hidden).squeeze(1)


class LinearResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        residual_scale: float = 0.35,
    ):
        super().__init__()
        self.linear_head = nn.Linear(int(input_dim), 1)
        self.residual_scale = float(residual_scale)

        dims = [int(input_dim), *[int(x) for x in hidden_dims if int(x) > 0]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        last_dim = dims[-1]
        layers.append(nn.Linear(last_dim, 1))
        self.residual_head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        linear_logit = self.linear_head(features).squeeze(1)
        residual_logit = self.residual_head(features).squeeze(1)
        return linear_logit + self.residual_scale * residual_logit


class PairedHybridModel(nn.Module):
    def __init__(
        self,
        emg_channels: int,
        mech_channels: int,
        tabular_dim: int,
        emg_hidden: int,
        mech_hidden: int,
        tabular_hidden: int,
        fusion_hidden: int,
        action_embedding_dim: int,
        dropout: float,
        subject_hidden: int,
        n_action_heads: int,
        biomarker_dim: int,
        biomarker_hidden: int,
        subject_fusion_hidden: int,
        aux_hidden: int,
        ssc_biomarker_dim: int,
        vdj_biomarker_dim: int,
        ssc_slot_idx: int,
        vdj_slot_idx: int,
    ):
        super().__init__()
        self.trial_encoder = TrialEncoder(
            emg_channels=emg_channels,
            mech_channels=mech_channels,
            tabular_dim=tabular_dim,
            emg_hidden=emg_hidden,
            mech_hidden=mech_hidden,
            tabular_hidden=tabular_hidden,
            fusion_hidden=fusion_hidden,
            action_embedding_dim=action_embedding_dim,
            dropout=dropout,
        )
        self.n_action_heads = int(n_action_heads)
        self.slot_latent_dim = int(self.trial_encoder.output_dim)
        self.slot_scorer = nn.Sequential(
            nn.Linear(self.slot_latent_dim, subject_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(subject_hidden, 1),
        )
        self.biomarker_dim = int(biomarker_dim)
        self.ssc_slot_idx = int(ssc_slot_idx)
        self.vdj_slot_idx = int(vdj_slot_idx)
        self.ssc_biomarker_dim = int(ssc_biomarker_dim)
        self.vdj_biomarker_dim = int(vdj_biomarker_dim)

        self.biomarker_encoder = None
        self.biomarker_head = None
        self.fusion_head = None
        if self.biomarker_dim > 0:
            self.biomarker_encoder = nn.Sequential(
                nn.Linear(self.biomarker_dim, biomarker_hidden),
                nn.LayerNorm(biomarker_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(biomarker_hidden, biomarker_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.biomarker_head = nn.Linear(biomarker_hidden, 1)
            fusion_input_dim = self.slot_latent_dim * 4 + self.n_action_heads + 1 + biomarker_hidden + 1
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input_dim, subject_fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(subject_fusion_hidden, 1),
            )

        self.ssc_aux_head = None
        if self.ssc_biomarker_dim > 0:
            self.ssc_aux_head = nn.Sequential(
                nn.Linear(self.slot_latent_dim, aux_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden, self.ssc_biomarker_dim),
            )

        self.vdj_aux_head = None
        if self.vdj_biomarker_dim > 0:
            self.vdj_aux_head = nn.Sequential(
                nn.Linear(self.slot_latent_dim, aux_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden, self.vdj_biomarker_dim),
            )

    def forward_slot_logits(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
        biomarker: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if emg.ndim != 4 or mechanics.ndim != 4 or action.ndim != 2:
            raise ValueError("paired model expects emg/mechanics with shape [batch, slot, channel, time]")

        batch_size, n_slots = emg.shape[:2]
        flat_emg = emg.reshape(batch_size * n_slots, *emg.shape[2:])
        flat_mech = mechanics.reshape(batch_size * n_slots, *mechanics.shape[2:])
        flat_action = action.reshape(batch_size * n_slots)
        flat_tabular = None if tabular is None else tabular.reshape(batch_size * n_slots, tabular.shape[-1])

        flat_latent = self.trial_encoder(flat_emg, flat_mech, flat_action, flat_tabular)
        if n_slots != self.n_action_heads:
            raise ValueError(
                f"paired model expected {self.n_action_heads} action slots, but received {n_slots}"
            )
        slot_logits = self.slot_scorer(flat_latent).squeeze(-1)
        return slot_logits.reshape(batch_size, n_slots)

    def encode_slots(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if emg.ndim != 4 or mechanics.ndim != 4 or action.ndim != 2:
            raise ValueError("paired model expects emg/mechanics with shape [batch, slot, channel, time]")

        batch_size, n_slots = emg.shape[:2]
        flat_emg = emg.reshape(batch_size * n_slots, *emg.shape[2:])
        flat_mech = mechanics.reshape(batch_size * n_slots, *mechanics.shape[2:])
        flat_action = action.reshape(batch_size * n_slots)
        flat_tabular = None if tabular is None else tabular.reshape(batch_size * n_slots, tabular.shape[-1])
        flat_latent = self.trial_encoder(flat_emg, flat_mech, flat_action, flat_tabular)
        if n_slots != self.n_action_heads:
            raise ValueError(
                f"paired model expected {self.n_action_heads} action slots, but received {n_slots}"
            )
        slot_latent = flat_latent.reshape(batch_size, n_slots, -1)
        slot_logits = self.slot_scorer(flat_latent).squeeze(-1).reshape(batch_size, n_slots)
        return slot_latent, slot_logits

    def aggregate_slot_logits(
        self,
        slot_logits: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_mask is None:
            action_mask = torch.ones_like(slot_logits, dtype=slot_logits.dtype, device=slot_logits.device)
        else:
            action_mask = action_mask.to(device=slot_logits.device, dtype=slot_logits.dtype)

        masked_fill = torch.full_like(slot_logits, -1e9)
        subject_logit = torch.where(action_mask > 0, slot_logits, masked_fill).max(dim=1).values
        no_valid = action_mask.sum(dim=1) <= 0
        return torch.where(no_valid, torch.zeros_like(subject_logit), subject_logit)

    def forward_subject_details(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
        biomarker: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        slot_latent, slot_logits = self.encode_slots(emg, mechanics, action, tabular)
        raw_subject_logit = self.aggregate_slot_logits(slot_logits, action_mask)

        if self.biomarker_encoder is not None:
            if biomarker is None:
                raise ValueError("biomarker features are required for the configured subject hybrid model")
            biomarker_hidden = self.biomarker_encoder(biomarker)
            biomarker_logit = self.biomarker_head(biomarker_hidden).squeeze(1)
            z_ssc = slot_latent[:, self.ssc_slot_idx, :]
            z_vdj = slot_latent[:, self.vdj_slot_idx, :]
            z_diff = z_ssc - z_vdj
            z_mean = 0.5 * (z_ssc + z_vdj)
            fusion_input = torch.cat(
                [
                    z_ssc,
                    z_vdj,
                    z_diff,
                    z_mean,
                    slot_logits,
                    raw_subject_logit.unsqueeze(1),
                    biomarker_hidden,
                    biomarker_logit.unsqueeze(1),
                ],
                dim=1,
            )
            subject_logit = self.fusion_head(fusion_input).squeeze(1)
        else:
            biomarker_hidden = raw_subject_logit.new_zeros((raw_subject_logit.shape[0], 0))
            biomarker_logit = raw_subject_logit.new_zeros(raw_subject_logit.shape)
            subject_logit = raw_subject_logit

        biomarker_pred_ssc = (
            self.ssc_aux_head(slot_latent[:, self.ssc_slot_idx, :])
            if self.ssc_aux_head is not None
            else raw_subject_logit.new_zeros((raw_subject_logit.shape[0], 0))
        )
        biomarker_pred_vdj = (
            self.vdj_aux_head(slot_latent[:, self.vdj_slot_idx, :])
            if self.vdj_aux_head is not None
            else raw_subject_logit.new_zeros((raw_subject_logit.shape[0], 0))
        )
        return {
            "subject_logit": subject_logit,
            "raw_subject_logit": raw_subject_logit,
            "biomarker_logit": biomarker_logit,
            "slot_logits": slot_logits,
            "slot_latent": slot_latent,
            "biomarker_hidden": biomarker_hidden,
            "biomarker_pred_ssc": biomarker_pred_ssc,
            "biomarker_pred_vdj": biomarker_pred_vdj,
        }

    def forward(
        self,
        emg: torch.Tensor,
        mechanics: torch.Tensor,
        action: torch.Tensor,
        tabular: torch.Tensor | None = None,
        biomarker: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        details = self.forward_subject_details(emg, mechanics, action, tabular, biomarker, action_mask)
        return details["subject_logit"]
