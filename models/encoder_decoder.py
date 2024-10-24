"""Neural network architecture for the flow model."""
import torch
from torch import nn

from models.utils import get_index_embedding, get_time_embedding, calc_distogram
from models import ipa_pytorch
from data import utils as du
import torch.nn.init as init

class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg, watermark_encoding):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.embed_size = self._cfg.c_pos_emb
        self.watermark_encoding = watermark_encoding
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk 
            self.embed_size += self.c_s
        if self.watermark_encoding and self._cfg.embed_watermark:
            self.embed_size += self.c_s
            self.watermark_encoder = nn.Linear(self._cfg.watermark_emb, self.c_s)
        if self._cfg.embed_chain:
            self.embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(self.embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, mask, aatypes, watermark, chain_index=None):
        # s: [b]
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            
        if self.watermark_encoding and self._cfg.embed_watermark:
            input_feats.append(self.watermark_encoder(watermark.unsqueeze(1).repeat([1, num_res, 1])))
            
        if self._cfg.embed_chain:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=100
                )
            )

        return self.linear(torch.cat(input_feats, dim=-1))

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins 
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, p_mask):
        num_batch, num_res, _ = s.shape
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        pos = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(pos)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = torch.concat(
            [cross_node_feats, relpos_feats, dist_feats], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats

class Encoder(nn.Module):
    
    def __init__(self, model_conf, watermark_encoder=True):
        super(Encoder, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.watermark_encoder = watermark_encoder
        self.node_embedder = NodeEmbedder(model_conf.node_features, self.watermark_encoder)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)
        
        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
                
    
    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        aatypes = input_feats['aatype']
        trans = input_feats['trans_1']
        rotmats = input_feats['rotmats_1']
        if self.watermark_encoder:
            watermark = input_feats['watermark']
        else:
            watermark = None

        # Initialize node and edge embeddings
        init_node_embed = self.node_embedder(node_mask, aatypes, watermark)
        init_edge_embed = self.edge_embedder(init_node_embed, trans, edge_mask)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats, trans,)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
    
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats
        }

class Decoder(nn.Module):
    def __init__(self, model_conf, watermark_encoder=False):
        super(Decoder, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.watermark_encoder = watermark_encoder
        self.node_embedder_clean = NodeEmbedder(model_conf.node_features, watermark_encoding=False)
        self.edge_embedder_clean = EdgeEmbedder(model_conf.edge_features)

        self.code_extractor = nn.Linear(self._model_conf.node_embed_size, self._model_conf.watermark_emb)
        torch.nn.init.xavier_uniform_(self.code_extractor.weight)
        torch.nn.init.constant_(self.code_extractor.bias, 0.0)
        
        self.predictor = nn.ModuleDict()
        for b in range(2):
            self.predictor[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.predictor[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.predictor[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.predictor[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.predictor[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.predictor[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )


    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        trans_t = input_feats['pred_trans']
        rotmats_t = input_feats['pred_rotmats']
        aatypes = input_feats['aatype']

        # Initialize node and edge embeddings
        init_node_embed_clean = self.node_embedder_clean(node_mask, aatypes, None)
        init_edge_embed_clean = self.edge_embedder_clean(init_node_embed_clean, trans_t, edge_mask)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t,)
        
        # predictor
        curr_rigids = self.rigids_ang_to_nm(du.create_rigid(rotmats_t, trans_t,))
        init_node_embed_clean = init_node_embed_clean * node_mask[..., None]
        node_embed = init_node_embed_clean * node_mask[..., None]
        edge_embed = init_edge_embed_clean * edge_mask[..., None]
        for b in range(2):
            ipa_embed = self.predictor[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.predictor[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.predictor[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.predictor[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.predictor[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            
            if b < 1:
                edge_embed = self.predictor[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        pred_code = self.code_extractor(node_embed).mean(dim=1)
    
        return pred_code


        