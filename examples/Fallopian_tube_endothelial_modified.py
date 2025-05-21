import scanpy as sc
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import sklearn.preprocessing
import episcanpy as epi

# --- Add project root to path if necessary ---
# Adjust this path if your script is located elsewhere relative to the scglue package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import scglue
    from scglue.models.cglue_soe_ot import CGLUESOE_OT_Model, configure_dataset_cglue_soe
    from scglue.models.data import AnnDatasetWithLabels
except ImportError as e:
    print(f"Error importing modified scglue components: {e}")
    print("Please ensure the modified scglue package (with CGLUE-SOE-OT) is installed or accessible.")
    sys.exit(1)
except AttributeError as e:
    print(f"Error importing specific components (check file/class names): {e}")
    print("Ensure 'CGLUESOE_OT_Model' and 'configure_dataset_cglue_soe' exist in 'scglue.models.cglue_soe_ot'.")
    sys.exit(1)

# --- Configuration ---
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
SAVE_DIR = "cglue_soe_ot_Fallopian_tube_endothelial_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Load Your Real Data ---
print("Loading real data...")
adata_ATAC = sc.read("/data/cabins/sccancer/hhy/TI/Fallopian_tube_ATAC.h5ad")
adata_RNA = sc.read("/data/cabins/sccancer/hhy/TI/Fallopian_tube_RNA.h5ad")

adata_ATAC_NK = adata_ATAC[adata_ATAC.obs["cell_type"] == 'endothelial cell',:]
samples_to_keep = ['55-year-old stage', '61-year-old stage', '62-year-old stage', '65-year-old stage']
mask = adata_ATAC_NK.obs['development_stage'].isin(samples_to_keep)
adata_ATAC_NK_Age = adata_ATAC_NK[mask].copy()

adata_RNA_NK = adata_RNA[adata_RNA.obs["cell_type"] == 'endothelial cell',:]
mask = adata_RNA_NK.obs['development_stage'].isin(samples_to_keep)
adata_RNA_NK_Age = adata_RNA_NK[mask].copy()

replacement_map = {
    '55-year-old stage': '55',
    '61-year-old stage': '61',
    '62-year-old stage': '62',
    '65-year-old stage': '65'
}
adata_ATAC_NK_Age.obs['age'] = adata_ATAC_NK_Age.obs['development_stage'].replace(replacement_map)
adata_RNA_NK_Age.obs['age'] = adata_RNA_NK_Age.obs['development_stage'].replace(replacement_map)

sc.pp.filter_cells(adata_RNA_NK_Age, min_genes=100)
sc.pp.filter_genes(adata_RNA_NK_Age, min_cells=3)
adata_RNA_NK_Age.layers["counts"] = adata_RNA_NK_Age.X.copy()
sc.pp.normalize_total(adata_RNA_NK_Age)
sc.pp.log1p(adata_RNA_NK_Age)
sc.pp.highly_variable_genes(adata_RNA_NK_Age, n_top_genes=2000)
adata_RNA_NK_Age = adata_RNA_NK_Age[:,adata_RNA_NK_Age.var['highly_variable'] ==True]

fpeak=0.03
epi.pp.binarize(adata_ATAC_NK_Age)
epi.pp.filter_features(adata_ATAC_NK_Age, min_cells=np.ceil(adata_ATAC_NK_Age.shape[0])*fpeak)

atac_adata = adata_ATAC_NK_Age
rna_adata = adata_RNA_NK_Age
adatas = {"rna": rna_adata, "atac": atac_adata}

# --- 2. Preprocessing Steps ---
print("Preprocessing data...")
age_order = ['55', '61', '62','65']
label_col = 'age'
try:
    for k in adatas:
        if 'age' not in adatas[k].obs:
            raise ValueError(f"'age' column not found in adata '{k}'.")
        adatas[k].obs['age'] = pd.Categorical(
            adatas[k].obs['age'], categories=age_order, ordered=True
        )
        if adatas[k].obs['age'].isnull().any():
             print(f"Warning: Some 'age' values in adata '{k}' did not match the expected categories {age_order} and resulted in NaNs.")
except Exception as e:
    print(f"Error processing 'age' column: {e}")
    sys.exit(1)
atac_adata.var['highly_variable'] = True

# --- 3. Define Vertices (No Graph Construction Needed) ---
print("Defining feature vertices...")
vertices = pd.Index(rna_adata.var_names.tolist() + atac_adata.var_names.tolist()).unique()
print(f"Total unique features (vertices): {len(vertices)}")

# --- 4. Configure Datasets for CGLUE-SOE-OT ---
print("Configuring datasets for CGLUE-SOE-OT...")
try:
    configure_dataset_cglue_soe(
        adatas["rna"],
        prob_model="Normal",
        use_highly_variable=True,
        use_label=label_col,
        use_layer=None,
        use_rep=None,
        use_obs_names=False
    )
    configure_dataset_cglue_soe(
        adatas["atac"],
        prob_model="Bernoulli",
        use_highly_variable=True,
        use_label=label_col,
        use_layer=None,
        use_rep=None,
        use_obs_names=False
    )
    print("Datasets configured.")
except Exception as e:
    print(f"Error configuring datasets: {e}")
    sys.exit(1)

# --- 5. Initialize CGLUE-SOE-OT Model ---
print("Initializing CGLUE-SOE-OT model...")
try:
    model = CGLUESOE_OT_Model(
        adatas=adatas,
        x2u_h_depth=2,
        x2u_h_dim=256,
        u2x_h_depth=2,
        u2x_h_dim=256,
        dropout=0.2,
        vertices=vertices.tolist(),
        latent_dim=50,
        random_seed=SEED
    )
    print("Model initialized.")
except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)

# --- 6. Compile Model ---
print("Compiling model...")
Times = '1' # This variable is used for saving filenames, keep it for consistency
lam_kl=0.1
lam_data=1.0
lam_triplet=200      # Modified: Weight for triplet loss
lam_ot=5.0
triplet_margin=10   # Modified: Margin for triplet loss
ot_epsilon=0.1
ot_tau=1.0
ot_max_iter=100
max_epochs=300

try:
    model.compile(
        lr=1e-3,
        lam_kl=lam_kl,
        lam_data=lam_data,
        lam_triplet=lam_triplet,
        lam_ot=lam_ot,
        triplet_margin=triplet_margin,
        ot_epsilon=ot_epsilon,
        ot_tau=ot_tau,
        ot_max_iter=ot_max_iter,
    )
    print("Model compiled.")
except Exception as e:
    print(f"Error compiling model: {e}")
    sys.exit(1)

# --- 7. Train Model ---
print("Fitting model...")
try:
    model.fit(
        adatas=adatas,
        val_split=0.2,
        data_batch_size=256,
        max_epochs=max_epochs,
        patience=35,
        reduce_lr_patience=20,
        wait_n_lrs=3,
        num_workers=0,
        directory=SAVE_DIR
    )
    print("Model fitting finished.")
except Exception as e:
    print(f"An error occurred during model fitting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 8. Post-Training Analysis ---
print("Extracting embeddings...")
try:
    embedding_key = "X_cgluesoe_ot"
    rna_adata.obsm[embedding_key] = model.encode_data("rna", rna_adata)
    atac_adata.obsm[embedding_key] = model.encode_data("atac", atac_adata)

    feature_embeddings = model.get_feature_embeddings()
    feature_embeddings_df = pd.DataFrame(feature_embeddings, index=model.vertices)

    print("RNA embeddings shape:", rna_adata.obsm[embedding_key].shape)
    print("ATAC embeddings shape:", atac_adata.obsm[embedding_key].shape)
    print("Feature embeddings shape:", feature_embeddings_df.shape)

    final_model_path = os.path.join(SAVE_DIR, "cglue_soe_ot_final_model.dill")
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path)

    # --- Optional: Visualization (UMAP) ---
    print("Generating UMAP visualization...")
    all_embeddings = np.concatenate([rna_adata.obsm[embedding_key], atac_adata.obsm[embedding_key]], axis=0)
    all_obs = pd.concat([
        rna_adata.obs[[label_col]].assign(modality='rna'),
        atac_adata.obs[[label_col]].assign(modality='atac')
    ], axis=0)
    vis_obs_names = rna_adata.obs_names.tolist() + atac_adata.obs_names.tolist()
    if len(vis_obs_names) != len(all_obs):
         print(f"Warning: Length mismatch between obs_names ({len(vis_obs_names)}) and combined obs ({len(all_obs)}). Using default index.")
         vis_obs_names = pd.RangeIndex(start=0, stop=len(all_obs), step=1)

    vis_adata = ad.AnnData(all_embeddings, dtype=all_embeddings.dtype)
    vis_adata.obs = all_obs
    vis_adata.obs_names = vis_obs_names
    vis_adata.obsm['X_cgluesoe_ot'] = all_embeddings

    print("Running neighbors and UMAP...")
    sc.pp.neighbors(vis_adata, use_rep='X_cgluesoe_ot', n_neighbors=15, random_state=SEED)
    sc.tl.umap(vis_adata, min_dist=0.3, random_state=SEED)

    print("Plotting UMAP...")
    umap_save_path = f'_cglue_soe_ot_Fallopian_endothelial_{Times}_results.png'
    sc.pl.umap(
        vis_adata,
        color=['modality', label_col],
        title=['UMAP by Modality', f'UMAP by {label_col.capitalize()}'],
        save=umap_save_path,
        show=False
    )
    print(f"UMAP plots saved to figures/ directory ({umap_save_path}).")

except Exception as e:
    print(f"An error occurred during post-training analysis: {e}")
    import traceback
    traceback.print_exc()

print("Script finished.")

print("Plotting loss curves...")
try:
    # Call the plotting method to display directly
    model.plot_loss_curves(save_path=None) # Set save_path to None to display directly
except Exception as e:
    print(f"An error occurred during loss plotting: {e}")
    import traceback
    traceback.print_exc()

import csv
import pickle
hyperparameters = {
    "lam_kl": lam_kl,
    "lam_data": lam_data,
    "lam_triplet": lam_triplet,
    "lam_ot": lam_ot,
    "triplet_margin": triplet_margin,
    "ot_epsilon": ot_epsilon,
    "ot_tau": ot_tau,
    "ot_max_iter": ot_max_iter,
    "max_epochs": max_epochs
}

csv_filename = os.path.join(SAVE_DIR, "hyperparameters_endothelial.csv")

try:
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'ParameterName_{Times}', 'Value'])
        for name, value in hyperparameters.items():
            writer.writerow([name, value])
    print(f"超参数已成功保存到 {csv_filename}")
except IOError:
    print(f"错误：无法写入文件 {csv_filename}")
