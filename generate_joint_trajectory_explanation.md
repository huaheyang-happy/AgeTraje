# `generate_joint_trajectory` 函数算法实现思路与细节说明

本文档详细阐述了 `generate_joint_trajectory_from_model` 函数（在 `models/trajectory.py` 中实现，并通过 `CGLUESOE_OT_Model` 的 `generate_joint_trajectory` 方法调用）的算法实现思路、具体细节以及它与 Monocle3 轨迹推断方法的关系。

## 1. 算法实现思路

`generate_joint_trajectory_from_model` 函数旨在从多模态单细胞数据（例如 RNA-seq 和 ATAC-seq）中推断并可视化一个统一的细胞状态轨迹。其核心思路是：

1.  **多模态数据整合与潜在空间学习**：利用预训练的深度学习模型（`CGLUESOE_OT_Model`）将不同模态的原始数据映射到一个共享的、低维的联合潜在空间。这个潜在空间能够捕获不同模态之间的共同生物学变异。
2.  **联合降维与可视化**：在整合后的联合潜在空间上应用 UMAP 等降维技术，将高维潜在表示进一步映射到二维（或三维）空间，以便于可视化和轨迹构建。
3.  **基于标签的轨迹构建**：利用细胞的预定义标签（例如年龄组、时间点或细胞分化阶段），在联合降维空间中构建一个单向的、平滑的细胞轨迹。这个轨迹代表了细胞状态的连续演变路径。
4.  **轨迹信息回投**：将生成的联合轨迹信息（如每个细胞在轨迹上的相对位置）投影回各个原始模态的 AnnData 对象中，以便于后续的模态特异性分析。
5.  **多维度可视化**：提供丰富的可视化功能，不仅展示联合轨迹本身，还展示不同模态的细胞在联合空间中的分布，以及它们在轨迹上的投影。

## 2. 算法实现细节

`generate_joint_trajectory_from_model` 函数的实现步骤如下：

### 步骤 1: 从模型获取潜在表示

-   **输入**：一个训练好的 `CGLUESOE_OT_Model` 实例 (`model`) 和一个包含不同模态 AnnData 对象的字典 (`adatas`)，以及用于定义轨迹顺序的标签键 (`label_key`)。
-   **过程**：
    -   函数遍历 `adatas` 字典中的每个模态（例如 "rna", "atac"）。
    -   对于每个模态，调用 `model.encode_data(modality_key, adata)`。这个方法会利用 `CGLUESOE_OT_Model` 中对应模态的编码器（`x2u`）将原始数据（`adata.X`）和其对应的标签（`adata.obs[label_key]`）编码到模型的潜在空间中。
    -   `encode_data` 返回的是潜在表示的均值（`u_mean`），这些均值被收集到 `latent_embeddings` 字典中。
    -   同时，每个模态的细胞标签也被收集到 `labels` 字典中。
-   **输出**：`latent_embeddings` (各模态的潜在表示字典) 和 `labels` (各模态的标签字典)。

### 步骤 2: 合并多模态数据

-   **输入**：步骤 1 得到的 `latent_embeddings` 和 `labels`。
-   **过程**：
    -   将所有模态的潜在表示（`latent_embeddings` 中的 NumPy 数组）沿行方向（`axis=0`）拼接起来，形成一个大的 NumPy 数组 `all_latent`。
    -   类似地，将所有模态的标签（`labels` 中的 NumPy 数组）沿行方向拼接起来，形成 `all_labels`。
    -   为了区分不同模态的细胞，创建一个 `modality_labels` 数组，记录每个细胞来源于哪个模态。
    -   构建一个 `pandas.DataFrame` 作为联合 AnnData 的 `obs`（观察值）信息，其中包含 `label_key` 和 `modality` 列。
    -   生成唯一的 `joint_obs_names`，例如将原始细胞名称前缀加上模态名称（`"rna_cell1"`, `"atac_cell2"`）。
-   **输出**：`all_latent` (所有模态的联合潜在表示), `all_labels` (所有模态的联合标签), `joint_obs` (联合观察值 DataFrame), `joint_obs_names` (联合观察值名称)。

### 步骤 3: 生成联合 UMAP 嵌入

-   **输入**：`all_latent` 和 `all_labels`，以及 UMAP 参数 (`umap_params`)。
-   **过程**：
    -   如果 `generate_umap` 参数为 `True`（默认行为），则调用 `generate_umap_from_latent` 函数。
    -   `generate_umap_from_latent` 内部使用 `umap.UMAP` 库，以 `all_latent` 作为输入，根据 `umap_params`（包括 `n_neighbors`, `min_dist`, `metric`, `random_state` 等）计算出一个二维（或更高维）的 UMAP 嵌入 `joint_umap_embedding`。
    -   如果 `generate_umap` 为 `False`，函数会尝试从现有 AnnData 对象的 `obsm` 中获取嵌入，这通常用于已经有预计算 UMAP 的情况。
-   **输出**：`joint_umap_embedding` (所有模态细胞的联合 UMAP 嵌入)。

### 步骤 4: 创建联合 AnnData 对象

-   **输入**：`all_latent`, `joint_obs`, `joint_obs_names`, `joint_umap_embedding`。
-   **过程**：
    -   创建一个新的 `AnnData` 对象 `joint_adata`。
    -   `joint_adata.X` 被设置为 `all_latent`（即高维潜在表示）。
    -   `joint_adata.obs` 被设置为 `joint_obs`。
    -   `joint_adata.obs_names` 被设置为 `joint_obs_names`。
    -   `joint_adata.obsm[embedding_key]` 被设置为 `joint_umap_embedding`（即二维 UMAP 嵌入）。
-   **输出**：`joint_adata` (包含所有模态细胞的联合潜在表示和 UMAP 嵌入的 AnnData 对象)。

### 步骤 5: 在联合 UMAP 空间中生成轨迹

-   **输入**：`joint_umap_embedding` 和 `all_labels`。
-   **过程**：
    -   调用 `generate_trajectory_from_embeddings(joint_umap_embedding, all_labels)` 函数。
    -   这个函数的核心是 `Trajectory` 类：
        -   它首先计算每个唯一标签（例如，每个年龄组）在 `joint_umap_embedding` 空间中的中位数（`np.median`）作为该标签的“中心点”。
        -   这些有序的中心点构成了轨迹的“骨架”。
        -   `Trajectory` 类内部的 `get_equally_spaced_points` 方法会利用这些中心点，通过样条插值（如果中心点数量足够）或线性插值，生成轨迹上的一系列等距点，从而形成一条平滑的曲线。
-   **输出**：`trajectory` (一个 `Trajectory` 对象，代表了联合空间中的细胞轨迹)。

### 步骤 6: 将轨迹信息投影回各个模态

-   **输入**：原始的 `adatas` 字典，`joint_umap_embedding`，以及生成的 `trajectory` 对象。
-   **过程**：
    -   函数遍历原始的 `adatas` 字典中的每个模态。
    -   对于每个模态的 AnnData 对象：
        -   从 `joint_umap_embedding` 中提取对应模态的 UMAP 嵌入，并更新该模态 AnnData 对象的 `obsm[embedding_key]`。这意味着每个模态的 UMAP 坐标现在是其在联合空间中的投影。
        -   调用 `project_data_to_trajectory(adata, trajectory, embedding_key, 'trajectory_position')`。这个函数会计算该模态中每个细胞在 `trajectory` 上的相对位置（一个 0 到 1 之间的值，类似于伪时间），并将结果存储在 `adata.obs['trajectory_position']` 中。
-   **输出**：更新后的 `adatas` 字典（每个 AnnData 对象都包含了在联合空间中的 UMAP 嵌入和轨迹位置信息）。

### 步骤 7: 可视化联合轨迹

-   **输入**：`joint_adata`, `adatas`, `trajectory`，以及可视化参数。
-   **过程**：
    -   如果 `visualize` 参数为 `True`，函数会生成多张图：
        -   **联合空间总览图（按模态着色）**：使用 `visualize_trajectory_umap` 绘制 `joint_adata` 的 UMAP 嵌入，并根据 `modality` 列进行着色，同时叠加生成的 `trajectory`。这展示了不同模态的细胞在整合后的联合空间中的分布。
        -   **联合空间总览图（按年龄着色）**：类似地，绘制 `joint_adata` 的 UMAP 嵌入，但根据 `label_key`（例如年龄）进行着色，显示不同年龄组在联合空间中的分布。
        -   **各模态轨迹投影图**：对于 `adatas` 字典中的每个原始模态，再次调用 `visualize_trajectory_umap`。这次绘制的是该模态在联合空间中的 UMAP 投影，并根据 `label_key` 进行着色，同样叠加联合 `trajectory`。这有助于理解每个模态的细胞是如何沿着统一轨迹分布的。
    -   所有图都支持保存到指定路径。
-   **输出**：生成的图表（可选保存为文件）。

## 3. 与 Monocle3 的关系

`generate_joint_trajectory_from_model` 函数在设计理念上受到了 Monocle3 等单细胞轨迹推断工具的启发，但针对多模态数据整合和特定轨迹生成需求进行了定制和扩展。

**相似之处：**

1.  **核心目标**：两者都致力于从单细胞数据中推断细胞状态的连续轨迹，以揭示细胞分化、发育或疾病进展等动态生物学过程。
2.  **降维方法**：都广泛采用 UMAP 作为主要的降维工具，将高维单细胞数据映射到低维空间，以便于可视化和轨迹构建。
3.  **“伪时间”概念**：`project_data_to_trajectory` 函数计算的 `trajectory_position` 与 Monocle3 中的“伪时间”（pseudotime）概念相似，它量化了每个细胞在推断出的轨迹上的进展程度。
4.  **可视化**：都提供在低维嵌入空间中绘制细胞散点图和叠加轨迹线的功能，以直观展示细胞状态的连续性。

**主要区别与创新：**

1.  **多模态整合能力**：
    *   **`generate_joint_trajectory_from_model`**：其核心优势在于能够整合多模态数据（如 RNA 和 ATAC）。它依赖于 `CGLUESOE_OT_Model` 学习到的共享潜在空间，从而在统一的框架下分析不同模态的数据，这是 Monocle3 等传统单模态轨迹推断工具所不具备的。
    *   **Monocle3**：主要设计用于单模态数据（通常是单细胞 RNA-seq），虽然可以通过一些外部方法进行整合，但其核心算法并非原生支持多模态。

2.  **轨迹生成方法**：
    *   **`generate_joint_trajectory_from_model`**：采用了一种基于预定义有序标签（如年龄组）的轨迹构建方法。它首先计算每个标签组在 UMAP 空间中的中心点，然后通过这些有序的中心点进行样条插值（或线性插值）来生成平滑的轨迹。这种方法更侧重于在已知的有序状态之间构建连续路径。
    *   **Monocle3**：通常通过构建一个“主图”（principal graph，例如基于最小生成树或弹性主图算法）来捕捉细胞状态之间的复杂连接关系，然后沿着这个图对细胞进行伪时间排序。这种方法更适用于从头发现复杂的细胞分化路径。

3.  **模型驱动**：
    *   **`generate_joint_trajectory_from_model`**：轨迹的生成是建立在深度学习模型 `CGLUESOE_OT_Model` 学习到的联合潜在表示之上的。这意味着轨迹的质量和结构在很大程度上取决于 VAE 模型对多模态数据的整合和降维能力。
    *   **Monocle3**：通常直接在原始或预处理后的表达数据上进行降维和图构建，不依赖于复杂的深度学习模型来学习潜在空间。

4.  **实现语言**：
    *   **`generate_joint_trajectory_from_model`**：使用 Python 实现，与 Python 生态系统（如 `anndata`, `scanpy`, `pytorch`, `umap`）无缝集成。
    *   **Monocle3**：主要是一个 R 语言包。

综上所述，`generate_joint_trajectory_from_model` 是一个为多模态单细胞数据量身定制的轨迹推断方法。它借鉴了 Monocle3 等工具的核心思想，但在多模态整合、轨迹构建策略和底层模型驱动方面进行了创新，使其能够更好地处理和解释复杂的多组学数据。
