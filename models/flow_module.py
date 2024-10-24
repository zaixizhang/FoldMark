from typing import Any
import torch
from torch import nn
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel, MapperNet
from models.encoder_decoder import Encoder, Decoder
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
from collections import OrderedDict
from models.lora_utils import LoRALinearLayer, LoRAMultiheadAttention, LoRACompatibleLinear, LoRACompatibleMultiheadAttention, LoRALinearLayerforward, LoRAMultiHeadAttentionforward, LoRACompatibleLinearforward


def extract_coarse_keys(keys):
    coarse_keys = set()
    for key in keys:
        # Split the key by the delimiter and drop the last part (e.g., "weight", "bias")
        if key.split('.')[-1] == 'head_weights':
            continue
        elif key.split('.')[-2] == 'self_attn':
            coarse_key = '.'.join(key.split('.')[:-1])
        elif len(key.split('.'))>=3 and key.split('.')[-3] == 'self_attn':
            coarse_key = '.'.join(key.split('.')[:-2])
        else:
            coarse_key = '.'.join(key.split('.')[:-1])
        coarse_keys.add(coarse_key)
    return sorted(coarse_keys)

def add_lora(model, model_keys, rank):
    lora_layers_list = []
    lora_params = []  # List to store LoRA layer parameters
    lora_attn_params = []
    for key in model_keys:
        attn_processor = model
        for sub_key in key.split("."):
            attn_processor = getattr(attn_processor, sub_key)

        if isinstance(attn_processor, torch.nn.MultiheadAttention):
            embed_dim = attn_processor.embed_dim
            num_heads = attn_processor.num_heads
            lora = LoRAMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rank=rank,
            )
            compatible_layer = LoRACompatibleMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            compatible_layer.load_state_dict(attn_processor.state_dict(), strict=False)
            
            replace_module(model, key, compatible_layer)
            compatible_layer.set_lora_layer(lora)
            lora_layers_list.append((compatible_layer, lora))
            
            # Add LoRA parameters to the list
            lora_attn_params += list(lora.parameters())
            
        if isinstance(attn_processor, torch.nn.Linear):
            lora = LoRALinearLayer(
                in_features=attn_processor.in_features,
                out_features=attn_processor.out_features,
                rank=rank,
            )
            compatible_layer = LoRACompatibleLinear(
                    in_features=attn_processor.in_features,
                    out_features=attn_processor.out_features
                )
            compatible_layer.load_state_dict(attn_processor.state_dict(), strict=False)
            # Replace the original layer with the LoRA-compatible one
            replace_module(model, key, compatible_layer)
            compatible_layer.set_lora_layer(lora)
            lora_layers_list.append((compatible_layer, lora))
            
            # Add LoRA parameters to the list
            lora_params += list(lora.parameters())
        else:
            continue


    # model_lora_parameters = []
    # # Set LoRA layers
    # for target_module, lora_layer in lora_layers_list:
    #     model_lora_parameters.extend(lora_layer.parameters())
                
    return model, lora_params, lora_attn_params, lora_layers_list

def replace_module(model, target_name, new_module):
    """
    Replaces a module in a model by recursively traversing the model.
    
    Args:
        model: The model containing the layer to replace.
        target_name: The full name of the layer (e.g., "encoder.layer.0.attention").
        new_module: The new module to replace the old one.
    """
    components = target_name.split('.')
    sub_module = model
    for name in components[:-1]:
        sub_module = getattr(sub_module, name)
    
    # Replace the layer with the new module
    setattr(sub_module, components[-1], new_module)


class ModelWithScaleWrapper(nn.Module):
    def __init__(self, model, lora_layers, rank=4):
        super().__init__()
        self.model = model
        self.lora_layers = lora_layers
        self.rank = rank

    def forward(self, hidden_states, scale_tensor=None):
        if scale_tensor is None:
            scale_tensor = torch.zeros(self.rank).to(hidden_states['res_mask'].device)
        for layer, _ in self.lora_layers:
            layer.set_scale_tensor(scale_tensor)
        return self.model(hidden_states)


class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self._encoder_decoder_cfg = cfg.encoder_decoder
        self.rank = cfg.encoder_decoder.rank

        # Set-up vector field prediction model
        self.mapper = MapperNet(input_size=self._encoder_decoder_cfg.watermark_emb, output_size=self.rank)
        model = FlowModel(cfg.model)
        state_dict = torch.load(self._exp_cfg.load_ckpt1)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('model.', '', 1) 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.requires_grad_(False)
        self.mapper.requires_grad_(False)
        
        model_keys = list(model.state_dict().keys())
        coarse_model_keys = extract_coarse_keys(model_keys)
        model, self.model_lora_parameters, self.model_lora_attn_parameters, self.lora_layers_list = add_lora(model, coarse_model_keys, self.rank)
        for param in self.model_lora_parameters:
            param.requires_grad_(True)
        for param in self.model_lora_attn_parameters:
            param.requires_grad_(False)
            
        self.model = ModelWithScaleWrapper(model, self.lora_layers_list, self.rank)
        
        self.encoder = Encoder(self._encoder_decoder_cfg)
        self.decoder = Decoder(self._encoder_decoder_cfg)
        state_dict = torch.load(self._exp_cfg.load_ckpt)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('encoder.', '', 1) 
            new_state_dict[name] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)
        self.decoder.load_state_dict(new_state_dict, strict=False)
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        scale = self.mapper(noisy_batch['watermark'][0:1])
        model_output = self.model(noisy_batch, scale)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        noisy_batch['pred_trans'], noisy_batch['pred_rotmats'] = model_output['pred_trans'], model_output['pred_rotmats']
        noisy_batch['pred_code'] = self.decoder(noisy_batch)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        
        gt_code = noisy_batch['watermark']
        pred_code = noisy_batch['pred_code']
        # Clamp logits to avoid extreme values leading to instability
        pred_code = torch.clamp(pred_code, min=-10, max=10)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        code_loss = loss_fn(pred_code, gt_code.float()).mean(dim=-1)
        #code_loss = (noisy_batch['t']>0.5).squeeze()*code_loss
        code_loss = noisy_batch['t'].squeeze()*code_loss
        # Add epsilon to avoid division by zero and NaN during mean computation
        epsilon = 1e-6
        predicted_classes = (torch.sigmoid(pred_code) > 0.5).float()
        # Use the epsilon to ensure no division by zero in recovery computation
        recovery = ((predicted_classes == gt_code.float()).float().mean(dim=-1)).clamp(min=epsilon)
        # Safeguard the final mean by adding epsilon
        recovery[noisy_batch['t'].squeeze()<0.5] = 1.
        recovery = recovery.mean().clamp(min=epsilon)
        
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "code_loss": code_loss,
            "recovery": recovery
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        batch['watermark'] = torch.randint(0, 2, (self._encoder_decoder_cfg.watermark_emb,)).repeat(num_batch, 1).float().to(batch['trans_1'].device)
        scale = self.mapper(batch['watermark'][0:1])
        
        samples, _, _, pred_trans_1, pred_rotmats_1 = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            scale
        )
        samples = samples[-1].numpy()
        batch['trans_1'] = pred_trans_1
        batch['rotmats_1'] = pred_rotmats_1
        encoder_output = self.encoder(batch)
        batch['pred_trans'], batch['pred_rotmats'] = encoder_output['pred_trans'], encoder_output['pred_rotmats']
        
        # batch['pred_trans'] = pred_trans_1
        # batch['pred_rotmats'] = pred_rotmats_1
        pred_code = self.decoder(batch)
        pred_code = torch.clamp(pred_code, min=-10, max=10)

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            predicted_classes = (torch.sigmoid(pred_code[i]) > 0.5).float()
            recovery = (predicted_classes == batch['watermark'][i].float()).float().mean().clamp(min=1e-6)
            batch_metrics.append(({'recovery': recovery}|mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        batch['watermark'] = torch.randint(0, 2, (self._model_cfg.watermark_emb,)).repeat(batch['res_mask'].shape[0], 1).float().to(batch['trans_1'].device)
        with torch.no_grad():
            encoder_output = self.encoder(batch)
            batch['trans_1'], batch['rotmats_1'] = encoder_output['pred_trans'], encoder_output['pred_rotmats']
       
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                scale = self.mapper(batch['watermark'][0:1])
                model_sc = self.model(noisy_batch, scale)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'code_loss' or 'recovery':
                continue
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
            total_losses[self._exp_cfg.training.loss]
            +  total_losses['auxiliary_loss'] + total_losses['code_loss']*2
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=list(self.model.parameters()) + list(self.decoder.parameters()),
            **self._exp_cfg.optimizer
        )


    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return
        atom37_traj, model_traj, _ = interpolant.sample(
            1, sample_length, self.model
        )

        os.makedirs(sample_dir, exist_ok=True)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0))
        _ = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
        )
