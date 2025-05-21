# -*- coding: utf-8 -*-
"""
Components specific to CGLUE-SOE model, now including OT loss
and removing graph/discriminator components.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

# Assuming base classes and utilities are available from the GLUE framework
# (Importing specific base class might be needed if glue.py is not directly usable)
# from . import glue # Example import
from ..num import EPS # Assuming EPS is defined in scglue.num

# --- Conditional Data Encoder (Kept from original CGLUE-SOE) ---

class ConditionalDataEncoder(nn.Module): # Inherit directly from nn.Module
    """
    Conditional Data Encoder q(u | x^(k), y^(k); phi^(k))
    Encodes data x conditioned on one-hot label y.

    Parameters
    ----------
    in_features
        Input data dimensionality
    label_dim
        Dimensionality of the one-hot encoded labels (number of classes)
    out_features
        Output latent dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """
    def __init__(
        self,
        in_features: int,
        label_dim: int, # Added label_dim
        out_features: int,
        h_depth: int = 2,
        h_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__() # Call nn.Module's init

        self.h_depth = h_depth
        # Input dimension is now data features + label dimension
        ptr_dim = in_features + label_dim

        layers = []
        for layer in range(self.h_depth):
            layers.append(nn.Linear(ptr_dim, h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.loc = nn.Linear(ptr_dim, out_features)
        self.std_lin = nn.Linear(ptr_dim, out_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> D.Normal:
        """
        Encode data x conditioned on label y.

        Parameters
        ----------
        x
            Input data tensor [batch_size, in_features]
        y
            One-hot encoded label tensor [batch_size, label_dim]

        Returns
        -------
        u
            Sample latent distribution (Normal)
        """
        # Concatenate data and labels
        xy = torch.cat([x, y], dim=1)

        # Pass through MLP layers
        ptr = self.hidden_layers(xy)

        # Compute latent distribution parameters
        loc = self.loc(ptr)
        # Ensure std is positive
        std = F.softplus(self.std_lin(ptr)) + EPS # Add small epsilon for numerical stability
        return D.Normal(loc, std)

# --- Triplet Loss Calculation (Kept from original CGLUE-SOE) ---

def calculate_triplet_loss(
    u_all: torch.Tensor,
    y_onehot_all: torch.Tensor,
    margin: float,
    device: torch.device
) -> torch.Tensor:
    """
    Calculates the Triplet Loss L_T based on CGLUE-SOE paper Eq. before (4).

    L_T = (1/C) * sum_{c=1}^{C-2} sum_{h=c+2}^{C} max(0, d(c, c+1) + delta - d(c, h))^2

    Parameters
    ----------
    u_all
        Tensor of all cell embeddings (means) in the batch [total_batch_size, z_dim]
    y_onehot_all
        Tensor of all one-hot encoded *ordered* labels [total_batch_size, num_classes]
    margin
        The margin delta for the triplet loss.
    device
        The torch device.

    Returns
    -------
    triplet_loss
        The calculated triplet loss value (scalar tensor).
    """
    num_classes = y_onehot_all.size(1)
    if num_classes < 3:
        # Triplet loss requires at least 3 classes
        return torch.tensor(0.0, device=device)

    centroids = []
    # Calculate centroids for each class present in the batch
    for c in range(num_classes):
        class_mask = y_onehot_all[:, c].bool()
        if class_mask.sum() == 0:
            # Handle cases where a class might not be in the current batch
            # Return 0 loss for this batch (simplest)
            # print(f"Warning: Class {c} not found in current batch for triplet loss.") # Optional warning
            return torch.tensor(0.0, device=device)
        centroids.append(u_all[class_mask].mean(dim=0))

    # This check might be redundant if we return 0 above, but good safeguard
    if len(centroids) != num_classes:
        # print(f"Warning: Number of centroids ({len(centroids)}) does not match num_classes ({num_classes}).")
        return torch.tensor(0.0, device=device)

    centroids = torch.stack(centroids) # [num_classes, z_dim]

    triplet_loss_sum = 0.0
    num_valid_triplets = 0

    # Iterate through triplets (c, c+1, h) where h >= c+2
    for c in range(num_classes - 2): # c from 0 to C-3 (inclusive) -> index for class 1 to C-2
        c_plus_1 = c + 1
        for h in range(c + 2, num_classes): # h from c+2 to C-1 (inclusive) -> index for class c+3 to C
            # Calculate squared Euclidean distances (more stable than sqrt for loss)
            # dist(a, b)^2 = ||a - b||^2
            d_c_cp1_sq = torch.sum((centroids[c] - centroids[c_plus_1])**2)
            d_c_h_sq = torch.sum((centroids[c] - centroids[h])**2)

            # Calculate the term inside max(0, ...)^2
            # Use sqrt for the comparison part as per the formula d(c,c+1) + delta - d(c,h)
            # Add small epsilon for stability if distances can be zero
            loss_term = torch.sqrt(d_c_cp1_sq + EPS) + margin - torch.sqrt(d_c_h_sq + EPS)

            # Apply max(0, ...) (hinge loss)
            squared_hinge_loss = F.relu(loss_term)
            triplet_loss_sum += squared_hinge_loss
            num_valid_triplets += 1

    # Average the loss over the number of valid triplets computed
    if num_valid_triplets > 0:
        triplet_loss = triplet_loss_sum / num_valid_triplets
    else:
        # This case might happen if num_classes is exactly 3, the loops won't run
        triplet_loss = torch.tensor(0.0, device=device)

    return triplet_loss

# --- Optimal Transport Loss Calculation (Based on uniPort Paper) ---

def cost_matrix(x_mu, x_std, y_mu, y_std, p=2):
    """
    Calculates the cost matrix between two sets of distributions (means and stds).
    Cost is based on Eq. 3 from uniPort paper: C_ij = ||mu_xi - mu_yj||^2 + ||sigma_xi - sigma_yj||^2
    Assumes diagonal covariance (using std vectors).

    Parameters
    ----------
    x_mu : torch.Tensor
        Means of the first set of distributions [batch_size_x, latent_dim]
    x_std : torch.Tensor
        Standard deviations of the first set of distributions [batch_size_x, latent_dim]
    y_mu : torch.Tensor
        Means of the second set of distributions [batch_size_y, latent_dim]
    y_std : torch.Tensor
        Standard deviations of the second set of distributions [batch_size_y, latent_dim]
    p : int, optional
        Power for the distance calculation (default is 2 for squared Euclidean)

    Returns
    -------
    torch.Tensor
        Cost matrix C [batch_size_x, batch_size_y]
    """
    # Expand dimensions for broadcasting:
    # x_mu: [Bx, 1, D], y_mu: [1, By, D] -> diff: [Bx, By, D]
    mu_diff = x_mu.unsqueeze(1) - y_mu.unsqueeze(0)
    std_diff = x_std.unsqueeze(1) - y_std.unsqueeze(0)

    # Calculate squared Euclidean distance (sum over latent dim)
    cost = torch.sum(mu_diff**p, dim=-1) + torch.sum(std_diff**p, dim=-1)
    return cost

def sinkhorn_knopp_unbalanced(
    C: torch.Tensor,
    epsilon: float,
    a: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 100,
    tau: float = 1.0, # Regularization for marginal constraints (uniPort Eq 4 D_KL terms)
    tol: float = 1e-3 # Convergence tolerance
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solves Unbalanced Optimal Transport using Sinkhorn-Knopp algorithm (IPOT variant).
    Based on uniPort paper Eq. 4 & 5.

    Parameters
    ----------
    C : torch.Tensor
        Cost matrix [batch_size_x, batch_size_y]
    epsilon : float
        Entropy regularization strength.
    a : torch.Tensor
        Source marginal distribution (histogram) [batch_size_x]
    b : torch.Tensor
        Target marginal distribution (histogram) [batch_size_y]
    max_iter : int, optional
        Maximum number of Sinkhorn iterations (default is 100).
    tau : float, optional
        Marginal constraint relaxation parameter (related to KL divergence weight).
        tau -> infinity recovers balanced OT. (default is 1.0).
    tol : float, optional
        Tolerance for checking convergence (default is 1e-3).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - Optimal Transport plan T [batch_size_x, batch_size_y]
        - Dual variable alpha [batch_size_x]
        - Dual variable beta [batch_size_y]
    """
    nx, ny = C.shape
    device = C.device

    # Initialize dual variables (beta is initialized first as per uniPort Eq. 5)
    beta = torch.ones(ny, device=device) / ny
    alpha = torch.ones(nx, device=device) / nx # Initialize alpha similarly

    # Kernel matrix
    K = torch.exp(-C / epsilon) + EPS # Add epsilon for stability

    fi = epsilon / (epsilon + tau) # Exponent for marginal constraints update

    for i in range(max_iter):
        alpha_prev = alpha.clone()
        beta_prev = beta.clone()

        # Update alpha (Eq. 5, first part)
        G_beta = K @ beta # G * beta^(l)
        alpha = (a / (G_beta + EPS))**fi * alpha**(tau / (epsilon + tau)) # Add EPS for stability

        # Update beta (Eq. 5, second part)
        GT_alpha = K.T @ alpha # G^T * alpha^(l+1) - Note: use updated alpha
        beta = (b / (GT_alpha + EPS))**fi * beta**(tau / (epsilon + tau)) # Add EPS for stability

        # Check for convergence (optional, but good practice)
        alpha_change = torch.norm(alpha - alpha_prev, p=1)
        beta_change = torch.norm(beta - beta_prev, p=1)
        if i > 10 and alpha_change < tol and beta_change < tol:
            # print(f"Sinkhorn converged at iteration {i}") # Optional log
            break

    # Calculate the optimal transport plan T* (uniPort Eq. 6)
    # T_ij = alpha_i * K_ij * beta_j
    T = alpha.unsqueeze(1) * K * beta.unsqueeze(0)

    return T, alpha, beta


def calculate_minibatch_uot_loss(
    u_dict: dict, # Dictionary of latent distributions {mod_name: D.Normal}
    epsilon: float = 0.1,
    max_iter: int = 100,
    tau: float = 1.0,
    reduction: str = 'mean' # How to aggregate loss across pairs
) -> torch.Tensor:
    """
    Calculates the Minibatch Unbalanced Optimal Transport loss between
    pairs of modalities based on uniPort paper Eq. 3-6.

    Parameters
    ----------
    u_dict : dict
        Dictionary where keys are modality names and values are the
        corresponding latent Normal distributions (output of ConditionalDataEncoder).
        Example: {'rna': Normal(loc_rna, std_rna), 'atac': Normal(loc_atac, std_atac)}
    epsilon : float, optional
        Entropy regularization for Sinkhorn (default is 0.1).
    max_iter : int, optional
        Max iterations for Sinkhorn (default is 100).
    tau : float, optional
        Marginal constraint relaxation parameter for UOT (default is 1.0).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied.
        'mean': the sum of the output will be divided by the number of pairs.
        'sum': the output will be summed. Default: 'mean'.


    Returns
    -------
    torch.Tensor
        The calculated Minibatch UOT loss (scalar tensor if reduction is 'mean' or 'sum').
    """
    modality_keys = list(u_dict.keys())
    num_modalities = len(modality_keys)
    if num_modalities < 2:
        # Need at least two modalities to calculate OT loss
        return torch.tensor(0.0, device=list(u_dict.values())[0].mean.device)

    total_ot_loss = 0.0
    num_pairs = 0

    # Iterate through all unique pairs of modalities
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            key1 = modality_keys[i]
            key2 = modality_keys[j]

            u1_dist = u_dict[key1]
            u2_dist = u_dict[key2]

            mu1, std1 = u1_dist.mean, u1_dist.stddev
            mu2, std2 = u2_dist.mean, u2_dist.stddev

            device = mu1.device
            batch_size1 = mu1.shape[0]
            batch_size2 = mu2.shape[0]

            if batch_size1 == 0 or batch_size2 == 0:
                continue # Skip if one batch is empty

            # Define marginal distributions (uniform for minibatch)
            a = torch.ones(batch_size1, device=device) / batch_size1
            b = torch.ones(batch_size2, device=device) / batch_size2

            # Calculate the cost matrix (Eq. 3)
            C = cost_matrix(mu1, std1, mu2, std2)

            # Solve UOT using Sinkhorn (Eq. 4 & 5)
            T_star, _, _ = sinkhorn_knopp_unbalanced(C, epsilon, a, b, max_iter, tau)

            # Calculate the UOT loss for this pair (Eq. 6)
            # loss = <C, T*> = sum(C * T*)
            pair_loss = torch.sum(C * T_star)
            total_ot_loss += pair_loss
            num_pairs += 1

    if num_pairs == 0:
        return torch.tensor(0.0, device=list(u_dict.values())[0].mean.device)

    # Apply reduction
    if reduction == 'mean':
        return total_ot_loss / num_pairs
    elif reduction == 'sum':
        return total_ot_loss
    elif reduction == 'none':
         # This case might be less useful here as we sum losses anyway
         # Returning the sum for now, could return a list of pair losses if needed
        return total_ot_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


__all__ = [
    "ConditionalDataEncoder",
    "calculate_triplet_loss",
    "calculate_minibatch_uot_loss"
]
