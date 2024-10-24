from typing import Any
import torch
from torch import nn
import time
import os
import random
import wandb
import copy
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
        self.encoder = Encoder(self._encoder_decoder_cfg)
        self.decoder = Decoder(self._encoder_decoder_cfg)
        state_dict = torch.load(self._exp_cfg.load_ckpt)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('encoder.', '', 1) 
            new_state_dict[name] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)
        self.decoder.load_state_dict(new_state_dict, strict=False)
        self.encoder.requires_grad_(True)
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
        gt_code = noisy_batch['watermark']

        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
        
        # Model output predictions.
        encoder_output = self.encoder(noisy_batch)
        pred_trans_1 = encoder_output['pred_trans']
        pred_rotmats_1 = encoder_output['pred_rotmats']
        noisy_batch['pred_trans'] = pred_trans_1
        noisy_batch['pred_rotmats'] = pred_rotmats_1
        pred_code = self.decoder(noisy_batch)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale 
        pred_bb_atoms *= training_cfg.bb_atom_scale 
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf = so3_utils.calc_rot_vf(gt_rotmats_1, pred_rotmats_1)
        rots_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        se3_vf_loss = trans_loss + rots_loss
        auxiliary_loss = bb_atom_loss
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        
        # Clamp logits to avoid extreme values leading to instability
        pred_code = torch.clamp(pred_code, min=-10, max=10)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        code_loss = loss_fn(pred_code, gt_code.float()).mean(dim=-1)
        # Add epsilon to avoid division by zero and NaN during mean computation
        predicted_classes = (torch.sigmoid(pred_code) > 0.5).float()
        # Use the epsilon to ensure no division by zero in recovery computation
        recovery = ((predicted_classes == gt_code.float()).float().mean(dim=-1))
        # Safeguard the final mean by adding epsilon
        recovery = recovery.mean()
        
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "rots_loss": rots_loss,
            "code_loss": code_loss,
            "recovery": recovery,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        
        noisy_batch = self.prepare_noisy_batch(batch)
        gt_code = noisy_batch['watermark']
        with torch.no_grad():
            encoder_output = self.encoder(noisy_batch)
            pred_trans_1 = encoder_output['pred_trans']
            pred_rotmats_1 = encoder_output['pred_rotmats']
            noisy_batch['pred_trans'] = pred_trans_1
            noisy_batch['pred_rotmats'] = pred_rotmats_1
            pred_code = self.decoder(noisy_batch)
        
        atom_pos = self.transrot_tot_atom_pos(pred_trans_1, pred_rotmats_1, res_mask).numpy()

        batch_metrics = []
        for i in range(num_batch):
            # Write out sample to PDB file
            final_pos = atom_pos[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, wandb.Molecule(saved_path)]
                )

        predicted_classes = (torch.sigmoid(pred_code) > 0.5).float()
        recovery = (predicted_classes == gt_code.float()).float().mean()
            
        gt_bb_atoms = all_atom.to_atom37(noisy_batch['trans_1'], noisy_batch['rotmats_1'])[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
            
        loss_denom = torch.sum(res_mask, dim=-1) * 3
        rmsd = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * res_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom
        batch_metrics.append({'recovery': recovery, 'rmsd': rmsd})

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
    
    def prepare_noisy_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        noisy_batch['t'] = torch.ones(num_batch)[:, None]

        noisy_batch['trans_t'] = trans_1

        noisy_batch['rotmats_t'] = rotmats_1
        
        noisy_batch['watermark'] = torch.randint(0, 2, (self._model_cfg.watermark_emb,)).repeat(num_batch, 1).float().to(batch['trans_1'].device)
        return noisy_batch    
    
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "Protein"],
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

    def transrot_tot_atom_pos(self, trans, rots, res_mask):
        res_mask = res_mask.detach().cpu()
        num_batch = res_mask.shape[0]
        rigids = du.create_rigid(rots, trans)
        atom37 = all_atom.compute_backbone(
            rigids,
            torch.zeros(
                trans.shape[0],
                trans.shape[1],
                2,
                device=trans.device
            )
        )[0]
        atom37 = atom37.detach().cpu()
        batch_atom37 = []
        for i in range(num_batch):
            batch_atom37.append(du.adjust_oxygen_pos(atom37[i], res_mask[i]))
        return torch.stack(batch_atom37)

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.prepare_noisy_batch(batch)
    
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            if k=='recovery':
                bar=True
            else:
                bar=False
            self._log_scalar(
                f"train/{k}", v, prog_bar=bar, batch_size=num_batch)
    
        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
            total_losses['bb_atom_loss']
            +  2*total_losses['code_loss'] + total_losses['trans_loss'] + total_losses['rots_loss']
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=list(self.encoder.parameters()) + list(self.decoder.parameters()),
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
