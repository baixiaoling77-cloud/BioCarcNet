import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from biocarcnet.data import edge_get_atom_features_dim, edge_get_bond_features_dim


class Enhanced3DSpatialEncoder(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.distance_sigma = nn.Parameter(torch.ones(num_heads))
        self.distance_bins = nn.Parameter(torch.linspace(0, 10, 16), requires_grad=False)
        self.distance_emb = nn.Embedding(16, num_heads)

    def forward(self, atom_pos):
        dist_matrix = torch.cdist(atom_pos, atom_pos)
        sigma = self.distance_sigma.view(1, -1, 1, 1).abs() + 1e-6
        gaussian_bias = -torch.pow(dist_matrix.unsqueeze(1), 2) / (2 * sigma)

        dist_flat = dist_matrix.unsqueeze(-1)
        bins = self.distance_bins.view(1, 1, 1, -1)
        bin_indices = torch.argmin(torch.abs(dist_flat - bins), dim=-1)
        bin_emb = self.distance_emb(bin_indices)
        bin_bias = bin_emb.permute(0, 3, 1, 2)
        return gaussian_bias + bin_bias


class SharedQueryTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # The name is retained for published-checkpoint compatibility.
        self.cond_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, query_vec, padding_mask=None, attn_bias=None):
        attn_mask = None
        key_padding_mask = padding_mask
        if attn_bias is not None:
            bsz, n_heads, n_tokens, _ = attn_bias.shape
            attn_mask = attn_bias.reshape(bsz * n_heads, n_tokens, n_tokens).to(dtype=x.dtype)
            attn_mask = torch.nan_to_num(attn_mask, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_mask = attn_mask.clamp(min=-1e4, max=1e4)

            if padding_mask is not None:
                pad_mask = padding_mask.view(bsz, 1, 1, n_tokens).expand(
                    bsz, n_heads, n_tokens, n_tokens
                )
                pad_mask = pad_mask.reshape(bsz * n_heads, n_tokens, n_tokens)
                attn_mask = attn_mask.masked_fill(pad_mask, -1e4)
                key_padding_mask = None

        x_norm = self.norm1(x)
        self_attn_out, self_attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
        )
        x = x + self_attn_out

        query = query_vec.unsqueeze(1)
        x_norm2 = self.norm2(x)
        query_out, query_attn_weights = self.cond_cross_attn(
            query=query,
            key=x_norm2,
            value=x_norm2,
            key_padding_mask=padding_mask,
            need_weights=True,
        )

        query_broadcast = query_out.expand_as(x)
        gate_weight = self.gate(torch.cat([x, query_broadcast], dim=-1))
        x = x * gate_weight + query_broadcast * (1.0 - gate_weight)
        x = x + self.ffn(self.norm3(x))

        return x, {
            'self_attn': self_attn_weights,
            'query_attn': query_attn_weights,
            'gate': gate_weight,
        }


class MultiScaleBondEncoder(nn.Module):
    def __init__(self, bond_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.local_encoder = nn.Linear(bond_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn_norm1 = nn.LayerNorm(hidden_dim)
        self.gcn_norm2 = nn.LayerNorm(hidden_dim)

        self.global_transformer = SharedQueryTransformerBlock(hidden_dim, num_heads, dropout)

        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bond_batch, query_vec):
        local_feat = self.local_encoder(bond_batch.x)
        mid_feat = F.relu(self.gcn_norm1(self.gcn1(local_feat, bond_batch.edge_index)))
        mid_feat = F.relu(self.gcn_norm2(self.gcn2(mid_feat, bond_batch.edge_index)))

        bond_dense, bond_mask = to_dense_batch(mid_feat, bond_batch.batch)
        padding_mask = ~bond_mask
        global_feat, attn_info = self.global_transformer(
            bond_dense, query_vec, padding_mask=padding_mask
        )
        global_feat_flat = global_feat[bond_mask]

        multi_scale = torch.cat([local_feat, mid_feat, global_feat_flat], dim=-1)
        fused_feat = self.fusion(multi_scale)
        importance = self.importance_head(fused_feat)
        weighted_feat = fused_feat * importance

        return weighted_feat, bond_mask, {
            'importance': importance,
            'attn_info': attn_info,
        }


class FingerprintEncoder(nn.Module):
    def __init__(self, fpn_dim, hidden_dim, dropout):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Linear(fpn_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
        )

        # This context vector is the shared zero query. Parameter names are
        # retained for published-checkpoint compatibility.
        self.cond_scale = nn.Linear(hidden_dim, hidden_dim)
        self.cond_shift = nn.Linear(hidden_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, fpn, context_vec):
        fpn_reduced = self.reducer(fpn)
        scale = self.cond_scale(context_vec)
        shift = self.cond_shift(context_vec)
        fpn_modulated = scale * fpn_reduced + shift
        return fpn_modulated + self.encoder(fpn_modulated)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, num_branches=3):
        super().__init__()
        self.num_branches = num_branches
        self.post_sum_scale = 2.0

        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * num_branches, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_branches),
            nn.Softmax(dim=-1),
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, atom_feat, bond_feat, fpn_feat):
        all_feats = torch.cat([atom_feat, bond_feat, fpn_feat], dim=-1)
        branch_weights = self.weight_net(all_feats)

        weighted_atom = atom_feat * branch_weights[:, 0:1]
        weighted_bond = bond_feat * branch_weights[:, 1:2]
        weighted_fpn = fpn_feat * branch_weights[:, 2:3]
        fused = weighted_atom + weighted_bond + weighted_fpn

        fused = fused * self.post_sum_scale
        fused = fused + self.fusion_net(fused)
        return fused, branch_weights


class BioConditionMultiHeadPredictor(nn.Module):
    def __init__(self, hidden_dim, task_num, dropout=0.5, num_conditions=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_num = task_num
        self.num_conditions = num_conditions

        self.species_emb = nn.Embedding(2, hidden_dim)
        self.sex_emb = nn.Embedding(2, hidden_dim)
        self.interaction_emb = nn.Embedding(num_conditions, hidden_dim)
        self.condition_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.shared_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.condition_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, task_num),
            )
            for _ in range(num_conditions)
        ])

    def build_condition_embeddings(self, device):
        condition_ids = torch.arange(self.num_conditions, device=device)
        species_ids = torch.div(condition_ids, 2, rounding_mode='floor')
        sex_ids = condition_ids % 2

        species_effect = self.species_emb(species_ids)
        sex_effect = self.sex_emb(sex_ids)
        interaction_effect = self.interaction_emb(condition_ids)

        cond_raw = species_effect + sex_effect + interaction_effect
        cond_vec = cond_raw + self.condition_adapter(cond_raw)

        return cond_vec, {
            'species_effect': species_effect,
            'sex_effect': sex_effect,
            'interaction_effect': interaction_effect,
            'condition_ids': condition_ids,
            'species_ids': species_ids,
            'sex_ids': sex_ids,
        }

    def forward(self, fused_feat, condition_idx=None):
        batch_size, hidden_dim = fused_feat.shape
        device = fused_feat.device

        cond_vec, condition_components = self.build_condition_embeddings(device)
        mol_feat = fused_feat.unsqueeze(1).expand(batch_size, self.num_conditions, hidden_dim)
        cond_feat = cond_vec.unsqueeze(0).expand(batch_size, self.num_conditions, hidden_dim)

        gamma, beta = self.film(cond_vec).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(0)
        beta = beta.unsqueeze(0)
        film_feat = mol_feat * (1.0 + gamma) + beta

        gate_weight = self.gate(torch.cat([mol_feat, cond_feat], dim=-1))
        conditioned_feat = gate_weight * film_feat + (1.0 - gate_weight) * mol_feat
        conditioned_feat = conditioned_feat + self.shared_adapter(conditioned_feat)

        all_logits = torch.stack(
            [head(conditioned_feat[:, idx, :]) for idx, head in enumerate(self.condition_heads)],
            dim=1,
        )

        selected_logits = None
        if condition_idx is not None:
            batch_index = torch.arange(batch_size, device=device)
            selected_logits = all_logits[batch_index, condition_idx]

        selected_conditioned_feat = None
        if condition_idx is not None:
            batch_index = torch.arange(batch_size, device=device)
            selected_conditioned_feat = conditioned_feat[batch_index, condition_idx]

        return selected_logits, all_logits, {
            'condition_embeddings': cond_vec,
            'condition_gate': gate_weight,
            'condition_components': condition_components,
            'selected_conditioned_feat': selected_conditioned_feat,
        }


class BioCarcNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_size
        self.num_heads = 8
        self.dropout = args.dropout
        self.task_num = args.task_num
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.atom_dim = edge_get_atom_features_dim()
        self.bond_dim = edge_get_bond_features_dim()

        self.spatial_encoder = Enhanced3DSpatialEncoder(self.num_heads)
        self.atom_gcn = GCNConv(self.atom_dim - 3, self.hidden_dim)
        self.atom_transformer = SharedQueryTransformerBlock(
            self.hidden_dim, self.num_heads, self.dropout
        )
        self.bond_encoder = MultiScaleBondEncoder(
            self.bond_dim, self.hidden_dim, self.num_heads, self.dropout
        )
        self.fpn_encoder = FingerprintEncoder(2048, self.hidden_dim, self.dropout)
        self.feature_fusion = AdaptiveFeatureFusion(self.hidden_dim, num_branches=3)
        self.bio_predictor = BioConditionMultiHeadPredictor(
            self.hidden_dim,
            self.task_num,
            dropout=0.5,
            num_conditions=4,
        )

        self.to(self.device)

    def forward(self, atom_batch, bond_batch, fpn, species_idx, sex_idx, smiles=None):
        batch_size = species_idx.size(0)
        shared_query = torch.zeros(batch_size, self.hidden_dim, device=species_idx.device)

        pos = atom_batch.x[:, -3:]
        feats = atom_batch.x[:, :-3]

        atom_x = F.relu(self.atom_gcn(feats, atom_batch.edge_index))
        atom_dense, atom_mask = to_dense_batch(atom_x, atom_batch.batch)
        pos_dense, _ = to_dense_batch(pos, atom_batch.batch)
        padding_mask = ~atom_mask

        spatial_bias = self.spatial_encoder(pos_dense)

        atom_global_seq, atom_attn_info = self.atom_transformer(
            atom_dense,
            shared_query,
            padding_mask=padding_mask,
            attn_bias=spatial_bias,
        )
        mask_float = atom_mask.float().unsqueeze(-1)
        atom_ctx = (atom_global_seq * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)

        bond_feat, bond_mask, bond_info = self.bond_encoder(bond_batch, shared_query)
        bond_dense, _ = to_dense_batch(bond_feat, bond_batch.batch)
        mask_float_bond = bond_mask.float().unsqueeze(-1)
        bond_ctx = (bond_dense * mask_float_bond).sum(dim=1) / (mask_float_bond.sum(dim=1) + 1e-8)

        fpn_ctx = self.fpn_encoder(fpn, shared_query)
        fused_feat, branch_weights = self.feature_fusion(atom_ctx, bond_ctx, fpn_ctx)

        condition_idx = species_idx.long().clamp(0, 1) * 2 + sex_idx.long().clamp(0, 1)
        logits, all_condition_logits, bio_condition_info = self.bio_predictor(
            fused_feat, condition_idx
        )

        attention_info = {
            'atom_self_attn': atom_attn_info['self_attn'],
            'atom_query_attn': atom_attn_info['query_attn'],
            'atom_gate': atom_attn_info['gate'],
            'bond_importance': bond_info['importance'],
            'bond_attn_info': bond_info['attn_info'],
            'branch_weights': branch_weights,
            'all_condition_logits': all_condition_logits,
            'all_condition_probs': torch.sigmoid(all_condition_logits.detach()),
            'bio_condition_gate': bio_condition_info['condition_gate'],
            'bio_condition_embeddings': bio_condition_info['condition_embeddings'],
            'bio_condition_components': bio_condition_info['condition_components'],
            'selected_conditioned_feat': bio_condition_info['selected_conditioned_feat'],
            'fused_feat': fused_feat,
        }

        return logits, attention_info


