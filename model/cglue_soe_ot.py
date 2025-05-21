# -*- coding: utf-8 -*-
"""
CGLUE-SOE-OT Model: Combines Conditional VAE, Ordinal Embedding (Triplet Loss),
and Optimal Transport for multi-omics integration.
Removes Discriminator and Graph VAE components from original CGLUE-SOE.
Includes loss history tracking and plotting functionality.
"""

import itertools
import os
import copy
import inspect # Import inspect for signature checking
from typing import Any, Dict, List, Mapping, Optional, Tuple
from math import ceil
import scanpy as sc
import torch.nn as nn
import matplotlib.pyplot as plt
import pathlib # Import pathlib

import ignite
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData

# Import from GLUE framework (adjust paths as needed)
from ..utils import config, logged, AUTO
from .base import Model, Trainer, TrainingPlugin # Base classes
from .data import AnnDatasetWithLabels, DataLoader as SCGLUEDataLoader
from .nn import autodevice, get_default_numpy_dtype # Utilities
# Import ignite handlers directly for clarity
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, TerminateOnNan, Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Average
from .plugins import LRScheduler, Tensorboard # Keep custom/modified plugins if needed

# Import specific components needed (Decoders, Prior)
from .sc import (
    Prior,
    NormalDataDecoder, ZINDataDecoder, ZILNDataDecoder, NBDataDecoder, ZINBDataDecoder
)
# Import CGLUE-SOE-OT specific components
from .cglue_soe_components_ot import (
    ConditionalDataEncoder,
    calculate_triplet_loss,
    calculate_minibatch_uot_loss
)

# --- CGLUE-SOE-OT Network Definition ---

class CGLUESOE_OT_Network(nn.Module):
    """
    Network definition for CGLUE-SOE-OT.
    """
    def __init__(
        self,
        x2u: Mapping[str, ConditionalDataEncoder],
        u2x: Mapping[str, torch.nn.Module],
        prior: Prior,
        feature_embeddings: nn.Embedding,
        vertices: pd.Index,
    ) -> None:
        super().__init__()

        if not set(x2u.keys()) == set(u2x.keys()) != set():
             raise ValueError("`x2u` and `u2x` should share the same keys and non-empty!")
        self.keys = list(x2u.keys())

        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2x = torch.nn.ModuleDict(u2x)
        self.prior = prior
        self.feature_embeddings = feature_embeddings
        self.vertices = vertices

        self.device = autodevice()
        self.to(self.device)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device if hasattr(self, '_device') else torch.device('cpu')

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

    def forward(self) -> None:
        raise NotImplementedError("CGLUESOE_OT_Network does not implement a single forward pass.")


# --- CGLUE-SOE-OT Trainer Definition ---

CGLUE_SOE_OT_DataTensors = Tuple[
    Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor],
    torch.Tensor,
]

@logged
class CGLUESOE_OT_Trainer(Trainer):
    """
    Trainer for CGLUE-SOE-OT network. Includes loss history tracking.
    """
    def __init__(
        self,
        net: CGLUESOE_OT_Network,
        lam_data: float = 1.0, lam_kl: float = 1.0,
        lam_triplet: float = 1.0, lam_ot: float = 1.0,
        triplet_margin: float = 0.1, ot_epsilon: float = 0.1,
        ot_max_iter: int = 100, ot_tau: float = 1.0,
        min_adjacent_dist: float = 0.0, # New parameter for adjacent distance penalty
        modality_weight: Mapping[str, float] = None,
        optim: str = "Adam", lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(net)

        self.min_adjacent_dist = min_adjacent_dist # Store the new parameter

        self.required_losses = []
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.required_losses += ["triplet_loss", "ot_loss", "total_loss"]

        # Dynamically add age group distances to required_losses
        # Assuming labels_ordered is available via net.modalities
        first_modality_key = self.net.keys[0]
        if first_modality_key in self.net.modalities and "labels_ordered" in self.net.modalities[first_modality_key]:
            labels_ordered = self.net.modalities[first_modality_key]["labels_ordered"]
            num_classes = len(labels_ordered)
            for i in range(num_classes):
                label_i = labels_ordered[i]
                for j in range(i + 1, num_classes):
                    label_j = labels_ordered[j]
                    self.required_losses.append(f"dist_age_{label_i}_{label_j}")

        self.earlystop_loss = "total_loss" # Monitor total loss for early stopping

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_triplet = lam_triplet
        self.lam_ot = lam_ot
        self.triplet_margin = triplet_margin
        self.ot_epsilon = ot_epsilon
        self.ot_max_iter = ot_max_iter
        self.ot_tau = ot_tau

        if modality_weight is None:
             modality_weight = {k: 1.0 for k in self.net.keys}
        if not modality_weight:
             self.modality_weight = {}
        else:
             if min(modality_weight.values()) < 0:
                 raise ValueError("Modality weight must be non-negative!")
             normalizer = sum(modality_weight.values()) / len(modality_weight) if len(modality_weight) > 0 else 1.0
             self.modality_weight = {k: v / normalizer for k, v in modality_weight.items()}

        self.lr = lr
        self.optimizer = getattr(torch.optim, optim)(
            net.parameters(), lr=self.lr, **kwargs,
        )

        self.history = {'train': [], 'val': []}

    def format_data(self, data: List[torch.Tensor]) -> CGLUE_SOE_OT_DataTensors:
        """ Formats data from DataLoader. """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        num_expected_tensors = 5 * K + 1
        if len(data) != num_expected_tensors:
             if len(data) == num_expected_tensors + 3:
                 self.logger.warning("Ignoring unexpected graph tensors in format_data.")
                 data = data[:num_expected_tensors]
             else:
                 raise ValueError(f"Expected {num_expected_tensors} tensors, got {len(data)}")

        x, xrep, xbch, y_onehot, xdwt, pmsk = (
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], data[5*K]
        )

        x = {k: x[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xrep = {k: xrep[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xbch = {k: xbch[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        y_onehot = {k: y_onehot[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xdwt = {k: xdwt[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xflag = {k: torch.as_tensor(i, dtype=torch.long, device=device).expand(x[k].shape[0]) for i, k in enumerate(keys)}
        pmsk = pmsk.to(device, non_blocking=True)

        return x, xrep, xbch, y_onehot, xdwt, xflag, pmsk

    def compute_losses(self, data: CGLUE_SOE_OT_DataTensors) -> Mapping[str, torch.Tensor]:
        """ Compute loss functions for CGLUE-SOE-OT. """
        net = self.net
        x, xrep, xbch, y_onehot, xdwt, xflag, pmsk = data

        # --- Encoding ---
        u = {}
        for k in net.keys:
            encoder_input = xrep[k] if xrep[k].numel() else x[k]
            u[k] = net.x2u[k](encoder_input, y_onehot[k])
        usamp = {k: u[k].rsample() for k in net.keys}
        prior = net.prior()

        # --- VAE Loss (Reconstruction + KL) ---
        x_nll = {}
        x_kl = {}
        if not hasattr(net, 'feature_embeddings'):
             raise AttributeError("Network needs 'feature_embeddings' layer.")
        vsamp = net.feature_embeddings.weight

        for k in net.keys:
            try:
                feature_indices = getattr(net, f"{k}_idx")
            except AttributeError:
                raise AttributeError(f"Feature index buffer '{k}_idx' not found.")
            v_k = vsamp[feature_indices]

            decoder_instance = net.u2x[k]
            l_k = None
            if isinstance(decoder_instance, (NBDataDecoder, ZINBDataDecoder)):
                l_k = x[k].sum(dim=1, keepdim=True)
                l_k = torch.max(l_k, torch.tensor(1.0, device=l_k.device))

            recon_dist = decoder_instance(usamp[k], v_k, xbch[k], l_k)
            x_nll[k] = -recon_dist.log_prob(x[k]).sum(dim=1).mean()
            x_kl[k] = D.kl_divergence(u[k], prior).sum(dim=1).mean()

        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        weighted_x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        # --- Triplet Loss ---
        u_mean_all = torch.cat([u[k].mean for k in net.keys], dim=0)
        y_onehot_all = torch.cat([y_onehot[k] for k in net.keys], dim=0)
        triplet_loss = calculate_triplet_loss(u_mean_all, y_onehot_all, self.triplet_margin, net.device, self.min_adjacent_dist)

        # --- Optimal Transport Loss ---
        ot_loss = calculate_minibatch_uot_loss(
            u, epsilon=self.ot_epsilon, max_iter=self.ot_max_iter, tau=self.ot_tau, reduction='mean'
        )

        # --- Combine Losses ---
        total_loss = (
            self.lam_data * weighted_x_elbo_sum
            + self.lam_triplet * triplet_loss
            + self.lam_ot * ot_loss
        )

        losses = {"triplet_loss": triplet_loss, "ot_loss": ot_loss, "total_loss": total_loss}
        for k in net.keys:
            losses.update({f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]})

        # --- Calculate Age Group Distances ---
        first_modality_key = net.keys[0]
        labels_ordered = net.modalities[first_modality_key]["labels_ordered"]
        num_classes = len(labels_ordered)

        batch_centroids = {}
        for i, label_str in enumerate(labels_ordered):
            class_mask = y_onehot_all[:, i].bool()
            if class_mask.sum() > 0:
                batch_centroids[label_str] = u_mean_all[class_mask].mean(dim=0)

        for i in range(num_classes):
            label_i = labels_ordered[i]
            if label_i not in batch_centroids:
                continue

            for j in range(i + 1, num_classes):
                label_j = labels_ordered[j]
                if label_j not in batch_centroids:
                    continue

                dist_sq = torch.sum((batch_centroids[label_i] - batch_centroids[label_j])**2)
                dist = torch.sqrt(dist_sq)

                losses[f"dist_age_{label_i}_{label_j}"] = dist

        return losses

    def train_step(self, engine: ignite.engine.Engine, data: List[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """ Performs a single training step. """
        self.net.train()
        formatted_data = self.format_data(data)
        losses = self.compute_losses(formatted_data)
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        return losses

    @torch.no_grad()
    def val_step(self, engine: ignite.engine.Engine, data: List[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """ Performs a single validation step. """
        self.net.eval()
        formatted_data = self.format_data(data)
        losses = self.compute_losses(formatted_data)
        return losses

    def fit(
        self, data: AnnDatasetWithLabels, val_split: float = 0.1,
        data_batch_size: int = 128, max_epochs: int = 500,
        patience: Optional[int] = 50, reduce_lr_patience: Optional[int] = 20,
        wait_n_lrs: Optional[int] = 3, random_seed: int = 0,
        directory: Optional[os.PathLike] = None,
        plugins: Optional[List[TrainingPlugin]] = None,
        num_workers: int = 0, pin_memory: bool = False, **kwargs,
    ) -> Dict[str, list]: # Modified return type
        """ Fit network using AnnDatasetWithLabels and track loss history. """
        if val_split <= 0 or val_split >= 1: raise ValueError("val_split must be between 0 and 1.")
        if data_batch_size <= 0: raise ValueError("data_batch_size must be positive.")

        self.history = {'train': [], 'val': []}

        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH)) if config.DATALOADER_FETCHES_PER_BATCH > 0 else data_batch_size
        data_train.getitem_size = getitem_size
        data_val.getitem_size = getitem_size
        data_train.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        data_val.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        train_loader = SCGLUEDataLoader(
            data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory and not config.CPU_ONLY,
            drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
            generator=torch.Generator().manual_seed(random_seed), persistent_workers=num_workers > 0,
        )
        val_loader = SCGLUEDataLoader(
            data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory and not config.CPU_ONLY,
            drop_last=False, generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=num_workers > 0,
        )

        # --- Setup Ignite Engines and Handlers ---
        train_engine = ignite.engine.Engine(self.train_step)
        val_engine = ignite.engine.Engine(self.val_step) if val_loader else None

        # Attach metrics to engines
        for item in self.required_losses:
            Average(output_transform=lambda output, item=item: output[item]).attach(train_engine, item)
            if val_engine:
                Average(output_transform=lambda output, item=item: output[item]).attach(val_engine, item)

        # Handler to log training metrics to history
        @train_engine.on(ignite.engine.Events.EPOCH_COMPLETED)
        def log_train_metrics(engine):
            metrics = engine.state.metrics
            self.history['train'].append({'epoch': engine.state.epoch, **metrics})
            if engine.state.epoch % config.PRINT_LOSS_INTERVAL == 0:
                 train_metrics_str = str({k: f"{v:.4f}" for k, v in metrics.items()})
                 log_msg = f"[Epoch {engine.state.epoch}] Train={train_metrics_str}"
                 if not val_engine: self.logger.info(log_msg)

        # Handler to run validation and log metrics
        if val_engine:
            @train_engine.on(ignite.engine.Events.EPOCH_COMPLETED)
            def validate_and_log(engine):
                val_engine.run(val_loader, max_epochs=1)
                metrics = val_engine.state.metrics
                self.history['val'].append({'epoch': engine.state.epoch, **metrics})
                if engine.state.epoch % config.PRINT_LOSS_INTERVAL == 0:
                     train_metrics_str = str({k: f"{v:.4f}" for k, v in engine.state.metrics.items()})
                     val_metrics_str = str({k: f"{v:.4f}" for k, v in metrics.items()})
                     self.logger.info(f"[Epoch {engine.state.epoch}] Train={train_metrics_str}, Val={val_metrics_str}")

        # Terminate on NaN
        train_engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, TerminateOnNan())

        # Exception handling
        @train_engine.on(ignite.engine.Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and config.ALLOW_TRAINING_INTERRUPTION:
                self.logger.info("Stopping training due to user interrupt...")
                engine.terminate()
            else: raise e

        # Setup plugins
        default_plugins = [Tensorboard()]
        if reduce_lr_patience is not None and reduce_lr_patience > 0:
            default_plugins.append(
                LRScheduler(self.optimizer, monitor=self.earlystop_loss, patience=reduce_lr_patience)
            )
        if patience is not None and patience > 0:
             score_engine = val_engine if val_engine else train_engine
             def score_function(engine_to_score):
                 if self.earlystop_loss not in engine_to_score.state.metrics:
                     self.logger.warning(f"Early stopping monitor '{self.earlystop_loss}' not found. Returning 0.")
                     return 0.0
                 return -engine_to_score.state.metrics[self.earlystop_loss]

             default_plugins.append(
                 IgniteEarlyStopping(
                     patience=patience,
                     score_function=score_function,
                     trainer=train_engine, # Correct argument name is 'trainer'
                     min_delta=1e-4
                     # Removed check_finite
                 )
             )

        plugins = default_plugins + (plugins or [])

        # Attach Checkpoint handler
        if directory:
            checkpoint_dir = pathlib.Path(directory)
            score_engine_for_ckpt = val_engine if val_engine else train_engine
            def score_fn_for_ckpt(engine):
                if self.earlystop_loss not in engine.state.metrics: return 0.0
                # Ensure score is float before formatting
                score_val = -engine.state.metrics[self.earlystop_loss]
                return float(score_val) if isinstance(score_val, (torch.Tensor, np.number)) else 0.0


            checkpoint_handler = Checkpoint(
                {"net": self.net, "trainer": self},
                DiskSaver(checkpoint_dir, atomic=True, create_dir=True, require_empty=False),
                score_function=score_fn_for_ckpt,
                # --- *** Corrected filename_pattern *** ---
                filename_pattern="checkpoint_{global_step}.pt", # Use epoch number
                n_saved=config.CHECKPOINT_SAVE_NUMBERS,
                global_step_transform=global_step_from_engine(train_engine),
                # score_name removed as it's not in the pattern
            )
            # Attach checkpoint handler to the engine providing the score
            score_engine_for_ckpt.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, checkpoint_handler)


        # Attach other plugins
        for plugin in plugins:
             plugin_directory = pathlib.Path(directory) if directory else None
             if hasattr(plugin, 'attach'):
                 plugin.attach(
                     net=self.net, trainer=self, train_engine=train_engine,
                     val_engine=val_engine, train_loader=train_loader,
                     val_loader=val_loader, directory=plugin_directory
                 )

        # Run training
        try:
            train_engine.run(train_loader, max_epochs=max_epochs)
        finally:
            data.clean()
            data_train.clean()
            data_val.clean()

        return self.history

    def get_losses(
        self, data: AnnDatasetWithLabels, data_batch_size: int = 128,
        random_seed: int = 0, num_workers: int = 0, pin_memory: bool = False
    ) -> Mapping[str, float]:
        """ Calculates losses on a given dataset. """
        getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH)) if config.DATALOADER_FETCHES_PER_BATCH > 0 else data_batch_size
        data.getitem_size = getitem_size
        data.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        loader = SCGLUEDataLoader(
            data, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory and not config.CPU_ONLY,
            drop_last=False, generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=num_workers > 0,
        )
        try:
            losses = super().get_losses(loader) # Calls base Trainer.get_losses
        finally:
            data.clean()
        return losses

    def state_dict(self) -> Mapping[str, Any]:
        state = {"optimizer": self.optimizer.state_dict()}
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict.pop("optimizer"))


# --- CGLUE-SOE-OT Model Public API ---

@logged
class CGLUESOE_OT_Model(Model):
    """
    Model class for CGLUE-SOE-OT. Includes loss plotting.
    """
    NET_TYPE = CGLUESOE_OT_Network
    TRAINER_TYPE = CGLUESOE_OT_Trainer

    MAX_EPOCH_PRG: float = 500.0
    PATIENCE_PRG: float = 50.0
    REDUCE_LR_PATIENCE_PRG: float = 20.0

    def __init__(
        self, adatas: Mapping[str, AnnData], vertices: List[str],
        latent_dim: int = 50, x2u_h_depth: int = 2, x2u_h_dim: int = 256,
        u2x_h_depth: int = 2, u2x_h_dim: int = 256, dropout: float = 0.2,
        random_seed: int = 0,
    ) -> None:
        """ Initialize CGLUE-SOE-OT model. """
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.vertices = pd.Index(vertices)
        self.modalities = {}
        x2u, u2x, idx = {}, {}, {}
        label_dim = None

        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(f"Dataset '{k}' not configured.")
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            self.modalities[k] = data_config

            if data_config.get("use_label") is None or data_config.get("label_dim") is None:
                 raise ValueError(f"Dataset '{k}' missing label configuration.")
            current_label_dim = data_config["label_dim"]
            if label_dim is None: label_dim = current_label_dim
            elif label_dim != current_label_dim: raise ValueError("Label dimensions mismatch.")

            features = data_config["features"]
            feature_idx = self.vertices.get_indexer(features)
            if (feature_idx < 0).any():
                 missing = self.vertices[feature_idx < 0]
                 raise ValueError(f"Dataset '{k}' features not in 'vertices': {missing.tolist()[:5]}...")
            idx[k] = torch.tensor(feature_idx, dtype=torch.long)

            x2u[k] = ConditionalDataEncoder(
                in_features=data_config["rep_dim"] or len(features), label_dim=label_dim,
                out_features=latent_dim, h_depth=x2u_h_depth, h_dim=x2u_h_dim, dropout=dropout,
            )

            decoder_cls = self._get_decoder_class(data_config["prob_model"])
            n_batches_dec = len(data_config["batches"]) if data_config.get("use_batch") and data_config.get("batches") is not None else 1

            sig = inspect.signature(decoder_cls.__init__)
            decoder_init_args = {"n_batches": n_batches_dec}
            if 'in_features' in sig.parameters: decoder_init_args["in_features"] = latent_dim
            if 'out_features' in sig.parameters: decoder_init_args["out_features"] = len(features)
            if 'h_dim' in sig.parameters: decoder_init_args['h_dim'] = u2x_h_dim
            if 'h_depth' in sig.parameters: decoder_init_args['h_depth'] = u2x_h_depth

            try: u2x[k] = decoder_cls(**decoder_init_args)
            except TypeError as e:
                 self.logger.error(f"Error initializing decoder {decoder_cls.__name__} for {k}: {e}")
                 raise e

        feature_embeddings = nn.Embedding(len(self.vertices), latent_dim)
        nn.init.xavier_uniform_(feature_embeddings.weight)
        prior = Prior()

        self._net = self.NET_TYPE(
             x2u=x2u, u2x=u2x, prior=prior,
             feature_embeddings=feature_embeddings, vertices=self.vertices
        )
        for k, v in idx.items():
             self._net.register_buffer(f"{k}_idx", v)
        self._trainer: Optional[Trainer] = None
        self.loss_history: Optional[Dict[str, list]] = None

    def _get_decoder_class(self, prob_model_name: str) -> type:
        """ Helper to get decoder class based on name. """
        mapping = {
            "Normal": NormalDataDecoder, "ZIN": ZINDataDecoder, "ZILN": ZILNDataDecoder,
            "NB": NBDataDecoder, "ZINB": ZINBDataDecoder, "Bernoulli": NormalDataDecoder,
        }
        if prob_model_name not in mapping:
            self.logger.warning(f"Decoder '{prob_model_name}' not mapped, using Normal.")
            return NormalDataDecoder
        return mapping[prob_model_name]

    def compile(
        self, lam_data: float = 1.0, lam_kl: float = 1.0, lam_triplet: float = 1.0,
        lam_ot: float = 1.0, triplet_margin: float = 0.1, ot_epsilon: float = 0.1,
        ot_max_iter: int = 100, ot_tau: float = 1.0,
        modality_weight: Optional[Mapping[str, float]] = None, lr: float = 1e-3, **kwargs,
    ) -> None:
        """ Compile the model for training. """
        if modality_weight is None: modality_weight = {k: 1.0 for k in self.modalities}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl, lam_triplet=lam_triplet, lam_ot=lam_ot,
            triplet_margin=triplet_margin, ot_epsilon=ot_epsilon, ot_max_iter=ot_max_iter,
            ot_tau=ot_tau, modality_weight=modality_weight, optim="Adam", lr=lr, **kwargs,
        )

    def fit(
        self, adatas: Mapping[str, AnnData], val_split: float = 0.1,
        data_batch_size: int = 128, max_epochs: int = AUTO,
        patience: Optional[int] = AUTO, reduce_lr_patience: Optional[int] = AUTO,
        wait_n_lrs: Optional[int] = 3, directory: Optional[os.PathLike] = None,
        num_workers: int = 0, pin_memory: bool = False, **kwargs
    ) -> None:
        """ Train the CGLUE-SOE-OT model and store loss history. """
        data = AnnDatasetWithLabels(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys], mode="train",
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size if data_batch_size > 0 else 1
        lr = self.trainer.lr if hasattr(self.trainer, 'lr') else 1e-3

        if max_epochs == AUTO:
            max_epochs = max(ceil(self.MAX_EPOCH_PRG / lr / batch_per_epoch), ceil(self.MAX_EPOCH_PRG)) if lr > 0 and batch_per_epoch > 0 else ceil(self.MAX_EPOCH_PRG)
            self.logger.info(f"Setting `max_epochs` = {max_epochs}")
        if patience == AUTO:
             patience = max(ceil(self.PATIENCE_PRG / lr / batch_per_epoch), ceil(self.PATIENCE_PRG)) if lr > 0 and batch_per_epoch > 0 else ceil(self.PATIENCE_PRG)
             self.logger.info(f"Setting `patience` = {patience}")
        elif patience is None: self.logger.info("Early stopping disabled.")
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(ceil(self.REDUCE_LR_PATIENCE_PRG / lr / batch_per_epoch), ceil(self.REDUCE_LR_PATIENCE_PRG)) if lr > 0 and batch_per_epoch > 0 else ceil(self.REDUCE_LR_PATIENCE_PRG)
            self.logger.info(f"Setting `reduce_lr_patience` = {reduce_lr_patience}")
        elif reduce_lr_patience is None: self.logger.info("LR scheduling disabled.")

        self.loss_history = self.trainer.fit(
            data=data, val_split=val_split, data_batch_size=data_batch_size,
            max_epochs=max_epochs, patience=patience, reduce_lr_patience=reduce_lr_patience,
            wait_n_lrs=wait_n_lrs, random_seed=self.random_seed, directory=directory,
            num_workers=num_workers, pin_memory=pin_memory, **kwargs
        )

    def plot_loss_curves(self, loss_keys: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
        """ Plots the training and validation loss curves. """
        if not hasattr(self, 'loss_history') or not self.loss_history:
            self.logger.warning("Loss history not available. Please run `fit` first.")
            return
        if not self.loss_history['train']:
            self.logger.warning("No training history found.")
            return

        train_hist_df = pd.DataFrame(self.loss_history['train']).set_index('epoch')
        val_hist_df = pd.DataFrame(self.loss_history['val']).set_index('epoch') if self.loss_history['val'] else None

        if loss_keys is None:
            loss_keys = [col for col in train_hist_df.columns if '_elbo' not in col and '_nll' not in col and '_kl' not in col]
            if not loss_keys: loss_keys = train_hist_df.columns.tolist()

        n_losses = len(loss_keys)
        n_cols = min(3, n_losses)
        n_rows = ceil(n_losses / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
        axes = axes.flatten()

        for i, key in enumerate(loss_keys):
            ax = axes[i]
            if key in train_hist_df:
                ax.plot(train_hist_df.index, train_hist_df[key], label=f"Train {key}", color='royalblue')
            if val_hist_df is not None and key in val_hist_df:
                ax.plot(val_hist_df.index, val_hist_df[key], label=f"Val {key}", color='darkorange', linestyle='--')

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{key.replace('_', ' ').title()} Curve")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        fig.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Loss curves saved to {save_path}")
            except Exception as e: self.logger.error(f"Failed to save loss curves: {e}")
            plt.close(fig)
        else: plt.show()

    @torch.no_grad()
    def encode_data(self, key: str, adata: AnnData, batch_size: int = 128) -> np.ndarray:
        """ Encode cell data from a specific modality into the latent space. """
        self.net.eval()
        encoder = self.net.x2u[key]
        if key not in self.modalities: raise ValueError(f"Modality key '{key}' not found.")
        data_config = self.modalities[key]

        dataset = AnnDatasetWithLabels([adata], [data_config], mode="eval", getitem_size=batch_size)
        dataloader = SCGLUEDataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
            collate_fn=SCGLUEDataLoader._collate
        )

        result = []
        for batch in dataloader:
            x, xrep, _, y_onehot, _, _ = batch
            x, xrep, y_onehot = x.to(self.net.device), xrep.to(self.net.device), y_onehot.to(self.net.device)
            encoder_input = xrep if xrep.numel() else x
            u_dist = encoder(encoder_input, y_onehot)
            result.append(u_dist.mean.detach().cpu())

        if not result:
             latent_dim = self.net.x2u[key].loc.out_features
             return np.empty((0, latent_dim), dtype=get_default_numpy_dtype())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def get_feature_embeddings(self) -> np.ndarray:
        """ Returns the learned feature embeddings. """
        if not hasattr(self.net, 'feature_embeddings'): raise AttributeError("Feature embedding layer not found.")
        self.net.eval()
        return self.net.feature_embeddings.weight.detach().cpu().numpy()

    def save(self, fname: os.PathLike) -> None:
        """ Save the CGLUE-SOE-OT model state. """
        model_state = {'net': self.net.state_dict()}
        if hasattr(self, '_trainer') and self._trainer:
             model_state['trainer'] = self.trainer.state_dict()
        if hasattr(self, 'loss_history') and self.loss_history:
             model_state['loss_history'] = self.loss_history
        model_state['vertices'] = self.vertices
        model_state['modalities'] = self.modalities
        model_state['init_args'] = self._get_init_args()

        try:
            torch.save(model_state, fname)
            self.logger.info(f"Model state saved to {fname}")
        except Exception as e: self.logger.error(f"Failed to save model state: {e}")

    @classmethod
    def load(cls, fname: os.PathLike, map_location=None) -> 'CGLUESOE_OT_Model':
        """ Load a saved CGLUE-SOE-OT model state. """
        state = torch.load(fname, map_location=map_location)
        init_args = state.get('init_args')
        if init_args is None: raise ValueError("Saved state missing init_args.")

        dummy_adatas = {}
        for k, mod_config in state['modalities'].items():
             dummy_adata = AnnData(np.empty((1, len(mod_config['features'])), dtype=np.float32))
             dummy_adata.var_names = mod_config['features']
             dummy_adata.obs[mod_config['use_label']] = pd.Categorical(
                 [mod_config['labels_ordered'][0]], categories=mod_config['labels_ordered'], ordered=True
             )
             if mod_config.get('use_batch'):
                 dummy_adata.obs[mod_config['use_batch']] = pd.Categorical(
                     [mod_config['batches'][0]], categories=mod_config['batches']
                 )
             dummy_adata.uns[config.ANNDATA_KEY] = mod_config
             dummy_adatas[k] = dummy_adata

        init_args['adatas'] = dummy_adatas
        init_args['vertices'] = state['vertices'].tolist()

        model = cls(**init_args)
        model.net.load_state_dict(state['net'])
        model.loss_history = state.get('loss_history')

        if 'trainer' in state and hasattr(model, '_trainer') and model._trainer:
             try:
                 compile_args = model._get_compile_args_from_trainer(model.trainer)
                 model.compile(**compile_args)
                 model.trainer.load_state_dict(state['trainer'])
             except Exception as e: model.logger.warning(f"Could not load trainer state: {e}")

        model.logger.info(f"Model loaded from {fname}")
        return model

    def _get_init_args(self) -> Dict[str, Any]:
         """ Helper to get arguments needed for __init__. """
         u2x_h_depth = None
         u2x_h_dim = None
         first_key = self.net.keys[0]
         if hasattr(self.net.u2x[first_key], 'hidden_layers') and self.net.u2x[first_key].hidden_layers:
             pass

         x2u_first = self.net.x2u[first_key]
         x2u_h_dim_infer = None
         dropout_infer = None
         if hasattr(x2u_first, 'hidden_layers') and len(x2u_first.hidden_layers) > 0 and isinstance(x2u_first.hidden_layers[0], nn.Linear):
             x2u_h_dim_infer = x2u_first.hidden_layers[0].out_features
         if hasattr(x2u_first, 'hidden_layers') and len(x2u_first.hidden_layers) > 3 and isinstance(x2u_first.hidden_layers[3], nn.Dropout):
             dropout_infer = x2u_first.hidden_layers[3].p

         return {
             'latent_dim': x2u_first.loc.out_features if hasattr(x2u_first, 'loc') else 50,
             'x2u_h_depth': x2u_first.h_depth if hasattr(x2u_first, 'h_depth') else 2,
             'x2u_h_dim': x2u_h_dim_infer or 256,
             'u2x_h_depth': u2x_h_depth or 2,
             'u2x_h_dim': u2x_h_dim or 256,
             'dropout': dropout_infer or 0.2,
             'random_seed': self.random_seed
         }

    def _get_compile_args_from_trainer(self, trainer_state_or_instance) -> Dict[str, Any]:
         """ Helper to get arguments for compile from trainer state or instance. """
         if isinstance(trainer_state_or_instance, dict):
             self.logger.warning("Cannot fully reconstruct compile args from saved trainer state. Using defaults.")
             lr = trainer_state_or_instance.get('optimizer', {}).get('param_groups', [{}])[0].get('lr', 1e-3)
             return {'lr': lr}
         else:
             trainer = trainer_state_or_instance
             return {
                 'lam_data': trainer.lam_data, 'lam_kl': trainer.lam_kl,
                 'lam_triplet': trainer.lam_triplet, 'lam_ot': trainer.lam_ot,
                 'triplet_margin': trainer.triplet_margin, 'ot_epsilon': trainer.ot_epsilon,
                 'ot_max_iter': trainer.ot_max_iter, 'ot_tau': trainer.ot_tau,
                 'modality_weight': trainer.modality_weight, 'lr': trainer.lr
             }

    def __repr__(self) -> str:
        """ String representation of the model. """
        try: trainer_repr = repr(self.trainer)
        except (RuntimeError, AttributeError): trainer_repr = "<Trainer not compiled/initialized>"
        net_repr = repr(self.net) if hasattr(self, '_net') else "<Network not initialized>"
        return f"CGLUE-SOE-OT model:\n\n{net_repr}\n\n{trainer_repr}\n"

# --- Configuration Function --- (Keep as is)
def configure_dataset_cglue_soe(
    adata: AnnData, prob_model: str, use_highly_variable: bool = True,
    use_layer: Optional[str] = None, use_rep: Optional[str] = None,
    use_batch: Optional[str] = None, use_label: str = None,
    use_dsc_weight: Optional[str] = None, use_obs_names: bool = False,
) -> None:
    """ Configure dataset for CGLUE-SOE-OT model training. """
    if use_label is None or use_label not in adata.obs:
        raise ValueError("`use_label` must be specified and exist in adata.obs.")

    logger = CGLUESOE_OT_Model.logger
    if config.ANNDATA_KEY in adata.uns:
         logger.warning("`configure_dataset` already called. Overwriting!")

    data_config = {"prob_model": prob_model}

    if use_highly_variable:
        if "highly_variable" not in adata.var:
            logger.warning("'highly_variable' not found, attempting selection.")
            try:
                # Check if data seems suitable for HVG selection (e.g., counts)
                count_like = False
                if adata.X is not None:
                    if scipy.sparse.issparse(adata.X):
                         # Check a small subset for non-negativity and integer-like values
                         subset_idx = np.random.choice(adata.shape[0], size=min(100, adata.shape[0]), replace=False)
                         X_subset = adata.X[subset_idx, :].toarray()
                         count_like = np.all(X_subset >= 0) and np.all(np.mod(X_subset, 1) == 0)
                    else:
                         count_like = np.all(adata.X >= 0) and np.all(np.mod(adata.X, 1) == 0)

                if 'counts' in adata.layers:
                     logger.info("Using layer 'counts' for HVG selection.")
                     sc.pp.highly_variable_genes(adata, layer='counts', inplace=True)
                elif count_like and adata.X is not None:
                     logger.info("Using adata.X for HVG selection.")
                     sc.pp.highly_variable_genes(adata, inplace=True)
                else:
                     raise ValueError("Suitable counts data not found in adata.X or adata.layers['counts'] for HVG selection.")

                if "highly_variable" not in adata.var: raise ValueError("HVG selection failed.")
            except Exception as e: raise ValueError(f"Mark HVGs first or ensure selection works: {e}")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.loc[adata.var['highly_variable']].index.to_list()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_list()

    data_config["use_layer"] = use_layer
    data_config["use_rep"] = use_rep
    data_config["rep_dim"] = adata.obsm[use_rep].shape[1] if use_rep and use_rep in adata.obsm else None

    data_config["use_batch"] = use_batch
    data_config["batches"] = None
    if use_batch:
        if use_batch not in adata.obs: raise ValueError(f"Batch column '{use_batch}' not found.")
        adata.obs[use_batch] = adata.obs[use_batch].astype('category')
        data_config["batches"] = adata.obs[use_batch].cat.categories.to_list()

    data_config["use_label"] = use_label
    if not pd.api.types.is_categorical_dtype(adata.obs[use_label]) or not adata.obs[use_label].cat.ordered:
         logger.warning(f"Label column '{use_label}' not ordered categorical. Converting...")
         adata.obs[use_label] = adata.obs[use_label].astype('category')
         try: ordered_categories = sorted(adata.obs[use_label].cat.categories, key=float)
         except ValueError: ordered_categories = sorted(adata.obs[use_label].cat.categories)
         logger.info(f"Inferred order for '{use_label}': {ordered_categories}")
         adata.obs[use_label] = adata.obs[use_label].cat.reorder_categories(ordered_categories, ordered=True)

    data_config["labels_ordered"] = adata.obs[use_label].cat.categories.to_list()
    data_config["label_dim"] = len(data_config["labels_ordered"])
    data_config["use_label_encoder"] = True

    data_config["use_dsc_weight"] = use_dsc_weight
    data_config["use_obs_names"] = use_obs_names

    adata.uns[config.ANNDATA_KEY] = data_config
    logger.info(f"Dataset configured for CGLUE-SOE-OT using label '{use_label}'.")


__all__ = ["CGLUESOE_OT_Model", "configure_dataset_cglue_soe"]
