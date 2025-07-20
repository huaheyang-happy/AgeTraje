# 基于最优传输的多组学整合模型 (CGLUE-SOE-OT) 说明文档

## 1. 引言

本说明文档旨在详细阐述 `AgeTraje` 项目中 `CGLUE-SOE-OT` 模型如何利用最优传输（Optimal Transport, OT）技术实现多组学数据的有效整合。`CGLUE-SOE-OT` 模型结合了条件变分自编码器（Conditional VAE）、序数嵌入（Ordinal Embedding，通过三元组损失实现）和最优传输，旨在学习不同组学数据在共享潜在空间中的统一表示，并特别关注了时间或年龄等序数信息的整合。

与原始的 `CGLUE-SOE` 模型相比，`CGLUE-SOE-OT` 移除了判别器和图VAE组件，转而引入了最优传输损失，以更直接地对不同模态的潜在表示进行对齐。

## 2. 核心概念

### 2.1 最优传输 (Optimal Transport, OT)

最优传输是一种数学工具，用于计算将一个概率分布“转换”为另一个概率分布所需的最小“成本”。在多组学整合中，OT 被用于对齐不同组学模态（例如RNA-seq和ATAC-seq）在潜在空间中的数据分布。通过最小化传输成本，模型能够学习到一个共享的潜在空间，使得来自不同模态但对应相同生物状态（如同一细胞类型或年龄）的数据点彼此靠近。

`CGLUE-SOE-OT` 模型中，OT 损失的计算基于 `uniPort` 论文中的非平衡最优传输（Unbalanced Optimal Transport, UOT）框架，它允许源分布和目标分布的质量不完全匹配，这在处理真实生物数据时更为灵活。

### 2.2 三元组损失 (Triplet Loss)

三元组损失是一种度量学习（Metric Learning）技术，常用于学习一个嵌入空间，使得相似的样本彼此靠近，不相似的样本彼此远离。在 `CGLUE-SOE-OT` 中，三元组损失被用于强制模型学习到的潜在表示能够反映预定义的序数关系，例如细胞的年龄或发育阶段。

具体来说，对于三个样本 (锚点 A, 正样本 P, 负样本 N)，三元组损失的目标是使得锚点 A 与正样本 P 之间的距离小于锚点 A 与负样本 N 之间的距离，并保持一定的裕度（margin）。在处理序数信息时，例如年龄组 `i < j < k`，模型会强制 `i` 和 `j` 的潜在表示距离小于 `i` 和 `k` 的距离，同时 `j` 和 `k` 的距离也小于 `i` 和 `k` 的距离，从而在潜在空间中形成一个有序的排列。

## 3. 模型架构 (`CGLUESOE_OT_Network`)

`CGLUESOE_OT_Network` 是 `CGLUE-SOE-OT` 模型的核心神经网络结构，它继承自 `torch.nn.Module`。

```python
class CGLUESOE_OT_Network(nn.Module):
    def __init__(
        self,
        x2u: Mapping[str, ConditionalDataEncoder],
        u2x: Mapping[str, torch.nn.Module],
        prior: Prior,
        feature_embeddings: nn.Embedding,
        vertices: pd.Index,
    ) -> None:
        super().__init__()
        self.x2u = torch.nn.ModuleDict(x2u) # 条件数据编码器
        self.u2x = torch.nn.ModuleDict(u2x) # 数据解码器
        self.prior = prior # 潜在空间的先验分布
        self.feature_embeddings = feature_embeddings # 特征嵌入层
        self.vertices = vertices # 特征（基因、区域等）的索引
        self.keys = list(x2u.keys()) # 模态名称列表
```

*   **`x2u` (ConditionalDataEncoder)**: 这是一个 `torch.nn.ModuleDict`，包含了针对每个组学模态的条件数据编码器。每个编码器 (`ConditionalDataEncoder`) 负责将原始数据 `x` 和其对应的独热编码标签 `y` 映射到一个潜在分布（通常是正态分布 `D.Normal`）的参数（均值 `loc` 和标准差 `std`）。
    *   **`ConditionalDataEncoder` 函数意义**:
        *   **输入**: 原始数据 `x` (例如基因表达矩阵或ATAC-seq计数) 和独热编码的标签 `y` (例如细胞类型或年龄组)。
        *   **内部结构**: 包含一系列线性层、LeakyReLU激活函数、BatchNorm1d和Dropout层，用于提取特征。
        *   **输出**: 一个 `torch.distributions.Normal` 对象，代表了输入数据在潜在空间中的条件分布。其均值 `loc` 和标准差 `std` 由网络的最后两层计算得出。
        *   **联系**: `x2u` 是 VAE 编码器部分，负责将高维的组学数据压缩到低维的潜在空间 `u`，同时利用标签信息进行条件编码，确保不同模态在潜在空间中的表示能够被标签信息引导。

*   **`u2x` (Decoders)**: 同样是一个 `torch.nn.ModuleDict`，包含了针对每个组学模态的数据解码器。每个解码器负责从潜在空间中的表示 `u` 重建原始数据 `x`。根据不同的数据类型（例如计数数据、连续数据），会使用不同的解码器类，如 `NormalDataDecoder`, `ZINDataDecoder`, `NBDataDecoder`, `ZINBDataDecoder` 等。
    *   **函数意义**:
        *   **输入**: 潜在表示 `u` 和特征嵌入 `v_k`。
        *   **内部结构**: 通常包含线性层和激活函数，将潜在表示映射回原始数据空间。
        *   **输出**: 一个概率分布对象（如 `Normal`, `NegativeBinomial` 等），用于计算重建数据的负对数似然（NLL）损失。
        *   **联系**: `u2x` 是 VAE 解码器部分，负责从潜在空间重构原始数据，确保潜在表示能够捕获原始数据的关键信息。

*   **`prior` (Prior)**: 定义了潜在空间 `u` 的先验分布，通常是一个标准正态分布。
    *   **函数意义**: 提供一个正则化项，鼓励潜在表示 `u` 遵循一个简单的分布，防止过拟合。
    *   **联系**: 在 VAE 框架中，KL 散度损失用于衡量编码器输出的潜在分布与先验分布之间的差异。

*   **`feature_embeddings` (nn.Embedding)**: 一个嵌入层，用于存储每个特征（例如基因或染色质区域）在潜在空间中的嵌入向量。
    *   **函数意义**: 为每个特征提供一个可学习的表示，这些表示在解码器中与潜在表示 `u` 结合，用于重建原始数据。
    *   **联系**: 这种设计允许模型在潜在空间中同时考虑细胞（通过 `u`）和特征（通过 `feature_embeddings`）的信息，从而更好地捕捉数据结构。

## 4. 训练过程 (`CGLUESOE_OT_Trainer`)

`CGLUESOE_OT_Trainer` 负责模型的训练循环、损失计算和优化。

```python
@logged
class CGLUESOE_OT_Trainer(Trainer):
    def __init__(
        self,
        net: CGLUESOE_OT_Network,
        modalities_config: Mapping[str, Any],
        lam_data: float = 1.0, lam_kl: float = 1.0,
        lam_triplet: float = 1.0, lam_ot: float = 1.0,
        triplet_margin: float = 0.1, ot_epsilon: float = 0.1,
        ot_max_iter: int = 100, ot_tau: float = 1.0,
        min_adjacent_dist: float = 0.0,
        modality_weight: Mapping[str, float] = None,
        optim: str = "Adam", lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(net)
        # ... 初始化损失权重、优化器等 ...
```

*   **`format_data` 函数意义**:
    *   **输入**: 原始数据加载器输出的张量列表。
    *   **输出**: 格式化后的字典，包含不同模态的输入数据 `x`、表示数据 `xrep`、批次信息 `xbch`、独热编码标签 `y_onehot`、数据权重 `xdwt`、模态标志 `xflag` 和掩码 `pmsk`。
    *   **联系**: 这是数据预处理步骤，确保数据以正确的格式传递给损失计算函数。

*   **`compute_losses` 函数意义**:
    *   **输入**: 格式化后的数据张量。
    *   **输出**: 一个字典，包含所有计算出的损失值（例如 `x_k_nll`, `x_k_kl`, `triplet_loss`, `ot_loss`, `total_loss` 等）。
    *   **核心逻辑**:
        1.  **编码**: 通过 `net.x2u` 对每个模态的数据进行编码，得到潜在分布 `u`。
        2.  **VAE 损失**:
            *   **重建损失 (NLL)**: `x_nll[k] = -recon_dist.log_prob(x[k]).sum(dim=1).mean()`。衡量解码器从潜在空间重建原始数据的准确性。
            *   **KL 散度损失**: `x_kl[k] = D.kl_divergence(u[k], prior).sum(dim=1).mean()`。衡量编码器输出的潜在分布 `u` 与先验分布 `prior` 之间的差异。
            *   **ELBO**: `x_elbo[k] = x_nll[k] + self.lam_kl * x_kl[k]`。变分下界，是 VAE 的主要优化目标。
        3.  **三元组损失**: `triplet_loss = calculate_triplet_loss(...)`。
            *   **函数意义**: 强制具有序数关系的类别在潜在空间中保持正确的相对距离。
            *   **联系**: 确保模型学习到的潜在空间不仅能整合不同模态，还能保留和利用标签中的序数信息。
        4.  **最优传输损失**: `ot_loss = calculate_minibatch_uot_loss(...)`。
            *   **函数意义**: 对齐不同模态在潜在空间中的分布。
            *   **联系**: 这是实现多组学整合的关键，通过最小化不同模态潜在分布之间的传输成本，使它们在共享潜在空间中对齐。
        5.  **总损失**: `total_loss = lam_data * weighted_x_elbo_sum + lam_triplet * triplet_loss + lam_ot * ot_loss`。所有损失项的加权和。
        6.  **年龄组距离**: 额外计算并记录了不同年龄组（或其他序数标签）在潜在空间中质心之间的距离，用于监控和分析。
    *   **联系**: `compute_losses` 是训练的核心，它定义了模型的优化目标，通过平衡重建、正则化、序数关系和模态对齐等多个目标来学习鲁棒的潜在表示。

### 4.1 `calculate_triplet_loss` 详细说明

```python
def calculate_triplet_loss(
    u_all: torch.Tensor,
    y_onehot_all: torch.Tensor,
    margin: float,
    device: torch.device,
    min_adjacent_dist: float = 0.0 # 新增参数，用于相邻类别距离惩罚
) -> torch.Tensor:
    # ...
```

*   **输入**:
    *   `u_all`: 批次中所有细胞的潜在嵌入均值。
    *   `y_onehot_all`: 批次中所有细胞的独热编码标签（按序排列）。
    *   `margin`: 三元组损失的裕度。
    *   `device`: PyTorch设备。
    *   `min_adjacent_dist`: 相邻类别之间强制的最小距离。
*   **核心逻辑**:
    1.  **计算类别质心**: 对于批次中存在的每个类别，计算其所有样本潜在嵌入的均值，作为该类别的质心。
    2.  **计算所有质心间的两两距离**: 预先计算所有类别质心之间的欧氏距离，提高效率。
    3.  **构建三元组并计算损失**: 遍历所有可能的有序三元组 `(i, j, k)`，其中 `i < j < k`（例如，年龄组 1、2、3）。对于每个三元组，计算两个损失项：
        *   `loss_term1 = d_i_j + margin - d_i_k`: 强制 `i` 到 `j` 的距离加上裕度小于 `i` 到 `k` 的距离。
        *   `loss_term2 = d_j_k + margin - d_i_k`: 强制 `j` 到 `k` 的距离加上裕度小于 `i` 到 `k` 的距离。
        *   使用 `F.relu` 应用铰链损失（hinge loss），只惩罚违反约束的情况。
    4.  **相邻距离惩罚**: 如果 `min_adjacent_dist > 0`，则对所有相邻类别对 `(c, c+1)` 施加惩罚。如果 `c` 和 `c+1` 之间的距离小于 `min_adjacent_dist`，则会产生一个惩罚项 `F.relu(min_adjacent_dist - current_dist)`。这有助于防止相邻类别在潜在空间中过于紧密地聚集，从而更好地保持序数结构。
    5.  **总损失**: 三元组损失和相邻距离惩罚的加权和。
*   **函数联系**: `calculate_triplet_loss` 是 `compute_losses` 的一个子模块，它通过引入序数信息来指导潜在空间的学习，使其不仅能整合数据，还能反映数据内在的序数结构。

### 4.2 `calculate_minibatch_uot_loss` 详细说明

```python
def calculate_minibatch_uot_loss(
    u_dict: dict, # 包含不同模态潜在分布的字典
    epsilon: float = 0.1, # 熵正则化强度
    max_iter: int = 100, # Sinkhorn迭代次数
    tau: float = 1.0, # 边际约束松弛参数
    reduction: str = 'mean'
) -> torch.Tensor:
    # ...
```

*   **输入**:
    *   `u_dict`: 字典，键为模态名称，值为对应的潜在正态分布（`D.Normal` 对象）。
    *   `epsilon`: Sinkhorn 算法的熵正则化参数。
    *   `max_iter`: Sinkhorn 算法的最大迭代次数。
    *   `tau`: 非平衡最优传输的边际约束松弛参数。`tau` 越大，对边际约束的惩罚越强，越接近平衡最优传输。
    *   `reduction`: 损失聚合方式（'mean', 'sum', 'none'）。
*   **核心逻辑**:
    1.  **遍历模态对**: 对所有独特的模态对（例如 RNA 和 ATAC）进行迭代。
    2.  **获取潜在分布参数**: 从 `u_dict` 中提取每对模态的潜在分布的均值 (`mu`) 和标准差 (`std`)。
    3.  **定义边际分布**: 对于小批量数据，通常假设边际分布是均匀的。
    4.  **计算成本矩阵 (`cost_matrix`)**:
        ```python
        def cost_matrix(x_mu, x_std, y_mu, y_std, p=2):
            # ...
        ```
        *   **函数意义**: 计算将一个模态的潜在分布转换为另一个模态的潜在分布所需的“成本”。
        *   **核心逻辑**: 基于 `uniPort` 论文中的公式 (Eq. 3)，成本定义为两个分布的均值之间的欧氏距离平方加上标准差之间的欧氏距离平方。这使得成本函数同时考虑了潜在表示的中心位置和其不确定性。
        *   **联系**: 成本矩阵是 OT 问题的核心输入，它量化了从一个模态的每个数据点到另一个模态的每个数据点的转换成本。
    5.  **求解非平衡最优传输 (`sinkhorn_knopp_unbalanced`)**:
        ```python
        def sinkhorn_knopp_unbalanced(
            C: torch.Tensor, epsilon: float, a: torch.Tensor, b: torch.Tensor,
            max_iter: int = 100, tau: float = 1.0, tol: float = 1e-3
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # ...
        ```
        *   **函数意义**: 使用 Sinkhorn-Knopp 算法求解非平衡最优传输问题，得到最优传输计划 `T_star`。
        *   **核心逻辑**: 这是一个迭代算法，通过交替更新两个对偶变量 `alpha` 和 `beta` 来逼近最优传输计划。`fi` 参数用于控制边际约束的松弛程度。
        *   **联系**: 这是 OT 损失计算的核心算法，它找到了在给定成本矩阵和边际分布下，最小化传输成本的传输方案。
    6.  **计算当前模态对的 OT 损失**: `pair_loss = torch.sum(C * T_star)`。这是成本矩阵 `C` 和最优传输计划 `T_star` 的 Hadamard 积之和，代表了该模态对的最小传输成本。
    7.  **聚合总 OT 损失**: 将所有模态对的 `pair_loss` 累加，并根据 `reduction` 参数进行平均或求和。
*   **函数联系**: `calculate_minibatch_uot_loss` 是 `compute_losses` 的另一个关键子模块，它通过最优传输机制直接对齐不同模态的潜在分布，从而实现多组学数据的整合。

*   **`train_step` 和 `val_step` 函数意义**:
    *   **`train_step`**: 执行一个训练步骤，包括数据格式化、损失计算、梯度清零、反向传播和优化器更新。
    *   **`val_step`**: 执行一个验证步骤，与训练步骤类似，但不进行梯度计算和参数更新（`torch.no_grad()`）。
    *   **联系**: 这两个函数定义了模型在训练和验证阶段的单次迭代行为。

*   **`fit` 函数意义**:
    *   **输入**: `AnnDatasetWithLabels` 数据集、验证集分割比例、批次大小、最大 epoch 数、早停耐心、学习率衰减耐心等。
    *   **核心逻辑**:
        1.  将数据集分割为训练集和验证集。
        2.  创建 `SCGLUEDataLoader` 用于加载数据。
        3.  设置 Ignite 引擎 (`train_engine`, `val_engine`)，并附加损失指标 (`Average`)。
        4.  配置 Ignite 事件处理器，例如记录训练/验证指标、处理 NaN 值、早停 (`IgniteEarlyStopping`) 和学习率调度 (`LRScheduler`)。
        5.  如果指定了目录，则设置检查点 (`Checkpoint`) 以保存模型状态。
        6.  运行训练引擎 (`train_engine.run(...)`)。
    *   **输出**: 训练和验证过程中的损失历史记录。
    *   **联系**: `fit` 函数是整个训练流程的入口点，它协调了数据加载、模型训练、验证、监控和保存等所有方面。

## 5. 模型API (`CGLUESOE_OT_Model`)

`CGLUESOE_OT_Model` 是用户与 `CGLUE-SOE-OT` 模型交互的主要接口。

```python
@logged
class CGLUESOE_OT_Model(Model):
    NET_TYPE = CGLUESOE_OT_Network
    TRAINER_TYPE = CGLUESOE_OT_Trainer
    # ...
```

*   **`__init__` 函数意义**:
    *   **输入**: `adatas` (包含不同模态 `AnnData` 对象的字典)、`vertices` (所有特征的列表)、`latent_dim` (潜在空间维度) 等。
    *   **核心逻辑**:
        1.  初始化随机种子。
        2.  遍历 `adatas` 中的每个模态，根据其配置 (`adata.uns[config.ANNDATA_KEY]`) 创建对应的 `ConditionalDataEncoder` (`x2u`) 和解码器 (`u2x`)。
        3.  初始化 `feature_embeddings` 和 `prior`。
        4.  构建 `CGLUESOE_OT_Network` 实例 (`self._net`)。
        5.  将特征索引注册为网络的缓冲区。
    *   **联系**: 这是模型实例化的入口，负责设置模型的各个组件。

*   **`compile` 函数意义**:
    *   **输入**: 优化器参数等。
    *   **核心逻辑**: 创建 `CGLUESOE_OT_Trainer` 实例 (`self._trainer`)，并将模型的模态配置 (`self.modalities`) 传递给训练器。
    *   **联系**: 在训练之前调用，用于配置训练器，包括损失权重、优化器等。

*   **`fit` 函数意义**:
    *   **输入**: 数据集和训练参数。
    *   **核心逻辑**: 调用内部 `_trainer` 的 `fit` 方法来启动模型训练。
    *   **联系**: 用户通过此方法启动模型的训练过程。

*   **`encode_data` 函数意义**:
    *   **输入**: `modality_key` (模态名称) 和 `adata` (要编码的 `AnnData` 对象)。
    *   **核心逻辑**:
        1.  将模型设置为评估模式 (`self.net.eval()`)。
        2.  从 `adata` 中提取原始数据 `x` 和独热编码标签 `y_onehot`。
        3.  使用对应模态的 `self.net.x2u` 编码器将数据编码到潜在空间，并返回潜在分布的均值。
    *   **输出**: 编码后的潜在表示（NumPy 数组）。
    *   **联系**: 允许用户在模型训练完成后，将新的原始数据编码到学习到的共享潜在空间中，用于后续分析。

*   **`generate_trajectory` 和 `generate_joint_trajectory` 函数意义**:
    *   **输入**: RNA 和 ATAC 的 `AnnData` 对象、标签键、UMAP 参数、平滑因子等。
    *   **核心逻辑**:
        1.  将 RNA 和 ATAC 数据封装到字典中。
        2.  调用 `trajectory` 模块中的 `generate_trajectory_from_model` 或 `generate_joint_trajectory_from_model` 函数。这些函数利用模型学习到的潜在空间，结合 UMAP 降维和图学习算法（如 Monocle3 的 `learn_graph` 和 `order_cells`）来推断细胞轨迹。
        3.  可选地生成 UMAP 嵌入、平滑轨迹和可视化结果。
        4.  将生成的轨迹对象保存到模型实例中。
    *   **输出**: 更新后的 `AnnData` 字典、生成的 `Trajectory` 对象（`generate_joint_trajectory` 还会返回一个联合 `AnnData` 对象）。
    *   **联系**: 这些函数是模型应用层面的重要功能，它们利用模型学习到的整合潜在空间来执行下游的生物学分析，例如细胞轨迹推断，从而揭示细胞分化或发育过程中的动态变化。

## 6. 函数之间的联系

`CGLUE-SOE-OT` 模型通过以下方式将各个函数和模块联系起来，实现多组学整合：

1.  **数据编码与潜在空间学习**:
    *   `CGLUESOE_OT_Model` 在初始化时构建 `CGLUESOE_OT_Network`，其中包含 `x2u` (ConditionalDataEncoder) 和 `u2x` (Decoders)。
    *   `ConditionalDataEncoder` 将不同模态的原始数据（`x`）和其对应的标签（`y`）编码到共享的潜在空间 `u` 中。标签的引入使得编码过程是条件性的，有助于在潜在空间中区分不同类别的样本。
    *   `u2x` 解码器则负责从潜在空间 `u` 重建原始数据，确保潜在表示的有效性。

2.  **多目标损失优化**:
    *   `CGLUESOE_OT_Trainer` 的 `compute_losses` 函数是核心，它整合了多种损失来指导模型训练：
        *   **VAE 损失 (NLL + KL)**: 确保每个模态的潜在表示能够有效捕获其原始数据的信息，并使其分布接近先验。
        *   **三元组损失 (`calculate_triplet_loss`)**: 利用标签的序数信息，强制潜在空间中的类别质心保持正确的相对距离和顺序，从而在整合过程中保留生物学上的序数结构（例如年龄梯度）。`min_adjacent_dist` 参数进一步确保了相邻类别之间的区分度。
        *   **最优传输损失 (`calculate_minibatch_uot_loss`)**: 这是实现多组学整合的关键。它通过计算不同模态潜在分布之间的传输成本，并最小化这个成本，直接对齐不同模态的潜在表示。`cost_matrix` 定义了传输的成本，而 `sinkhorn_knopp_unbalanced` 则求解了最优传输计划。这使得来自不同模态但对应相同生物状态的细胞在共享潜在空间中彼此靠近。

3.  **训练与应用**:
    *   `CGLUESOE_OT_Model.compile` 方法将 `CGLUESOE_OT_Network` 与 `CGLUESOE_OT_Trainer` 关联起来，配置训练参数。
    *   `CGLUESOE_OT_Model.fit` 方法启动训练循环，`CGLUESOE_OT_Trainer` 在每个训练步骤中计算并优化总损失。
    *   训练完成后，`CGLUESOE_OT_Model.encode_data` 允许用户将新的原始数据映射到学习到的整合潜在空间中。
    *   `CGLUESOE_OT_Model.generate_trajectory` 和 `generate_joint_trajectory` 函数则利用这个整合的潜在空间进行下游分析，例如细胞轨迹推断，从而揭示多组学数据背后的生物学过程。

综上所述，`CGLUE-SOE-OT` 模型通过巧妙地结合条件 VAE、三元组损失和最优传输，构建了一个强大的框架，不仅能够有效地整合多组学数据，还能在潜在空间中编码和利用重要的序数生物学信息，为深入理解复杂的生物学过程提供了有力的工具。
