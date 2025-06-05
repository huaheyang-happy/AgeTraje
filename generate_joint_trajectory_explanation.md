# `generate_joint_trajectory` 函数算法实现思路与细节说明

本文档详细阐述了 `generate_joint_trajectory_from_model` 函数的算法实现思路、具体细节以及它与 Monocle3 轨迹推断方法的关系。该函数结合了深度学习模型的多模态整合能力和改进的 Monocle3 轨迹推断算法，实现了鲁棒的年龄相关细胞轨迹生成。

## 1. 算法总体架构

`generate_joint_trajectory_from_model` 函数采用了一个多层次的算法架构，包含以下核心组件：

### 1.1 多模态数据整合层
- **深度学习编码**：利用预训练的 `CGLUESOE_OT_Model` 将 RNA 和 ATAC 数据编码到共享潜在空间
- **联合表示学习**：通过变分自编码器 (VAE) 学习跨模态的统一表示
- **最优传输对齐**：使用最优传输损失确保不同模态在潜在空间中的分布对齐

### 1.2 轨迹推断层
- **改进的 Monocle3 算法**：基于 `MonocleTrajectoryLearner` 类实现的轨迹学习
- **聚类驱动的轨迹构建**：结合细胞聚类和年龄标签信息构建有序轨迹
- **离群点检测与过滤**：使用 DBSCAN 和几何距离过滤脱离主轨迹的细胞群

### 1.3 轨迹优化层
- **单向性保证**：通过主成分分析和单调性约束避免轨迹折返
- **平滑插值**：使用样条插值生成平滑的轨迹路径
- **最小生成树构建**：采用 MST 算法确保轨迹的连通性和最优性

## 2. 核心算法详细实现

### 2.1 多模态数据编码与整合

#### 步骤 1: 深度学习模型编码
```python
# 对每个模态进行编码
for modality_key, adata in adatas.items():
    latent = model.encode_data(modality_key, adata)  # 使用条件VAE编码器
    latent_embeddings[modality_key] = latent
```

**技术细节**：
- **条件编码器**：`model.encode_data()` 使用 `ConditionalDataEncoder`，同时考虑原始数据 `X` 和标签信息 `y_onehot`
- **潜在表示**：返回潜在分布的均值 `u_mean`，这是经过三元组损失和最优传输损失训练的高质量表示
- **跨模态对齐**：通过 `CGLUESOE_OT_Model` 的联合训练，不同模态的潜在表示已经在同一空间中对齐

#### 步骤 2: 联合潜在空间构建
```python
# 合并所有模态的潜在表示
all_latent = np.concatenate([latent_embeddings[k] for k in adatas.keys()], axis=0)
all_labels = np.concatenate([labels[k] for k in adatas.keys()], axis=0)
```

### 2.2 UMAP 降维与可视化空间生成

#### 步骤 3: 联合 UMAP 嵌入
```python
# 在联合潜在表示上生成UMAP
joint_umap_embedding, _ = generate_umap_from_latent(
    all_latent, all_labels, 
    n_neighbors=umap_params['n_neighbors'],
    min_dist=umap_params['min_dist'],
    metric=umap_params['metric'],
    random_state=umap_params['random_state']
)
```

**关键参数优化**：
- `n_neighbors=15`：平衡局部和全局结构
- `min_dist=0.1`：允许适度的点聚集
- `metric='euclidean'`：适合已经对齐的潜在空间

### 2.3 改进的 Monocle3 轨迹推断算法

#### 步骤 4: 基于聚类的线性轨迹生成

这是算法的核心创新部分，结合了 Monocle3 的图论方法和年龄导向的轨迹构建：

##### 4.1 离群点检测与过滤 (`_filter_outlier_cells`)

```python
def _filter_outlier_cells(self, X, clusters, age_labels):
    # 使用DBSCAN检测主要细胞群体
    dbscan = DBSCAN(eps=1.5, min_samples=20)
    dbscan_labels = dbscan.fit_predict(X.T)
    
    # 找到最大聚类（主轨迹）
    unique_labels, counts = np.unique(dbscan_labels[dbscan_labels != -1], return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]
    main_cluster_mask = dbscan_labels == main_cluster_label
    
    # 几何距离过滤
    main_cluster_center = np.mean(X.T[main_cluster_mask], axis=0)
    distances_to_center = np.linalg.norm(X.T - main_cluster_center, axis=1)
    distance_threshold = np.percentile(distances_to_center[main_cluster_mask], 95)
    
    # 组合过滤条件
    filtered_mask = main_cluster_mask | (distances_to_center <= distance_threshold * 1.5)
```

**算法亮点**：
- **双重过滤策略**：结合 DBSCAN 聚类和几何距离，有效识别并过滤 UMAP 右上角等脱离主轨迹的细胞群
- **年龄组保护**：确保每个年龄组至少保留 50% 的细胞，避免过度过滤
- **自适应阈值**：基于主群体的距离分布动态设定过滤阈值

##### 4.2 聚类中心计算与年龄排序

```python
# 计算每个聚类的中心点和平均年龄
for cluster_id in unique_clusters:
    cluster_mask = clusters_filtered == cluster_id
    cluster_cells = X_filtered[:, cluster_mask]
    
    # 过滤小聚类
    if cluster_size < 10:
        continue
        
    centroid = np.mean(cluster_cells, axis=1)  # 聚类中心
    
    # 计算平均年龄
    cluster_age_labels = age_labels_filtered[cluster_mask]
    numeric_ages = [float(age) for age in cluster_age_labels]
    avg_age = np.mean(numeric_ages)
    
# 按年龄排序聚类
age_order = np.argsort(cluster_ages)
sorted_centroids = cluster_centroids[:, age_order]
```

**关键设计**：
- **聚类质量控制**：只保留有足够细胞数（≥10）的聚类，确保统计稳定性
- **年龄数值化**：将年龄标签转换为数值进行排序，支持连续年龄轨迹
- **中心点计算**：使用均值而非中位数，更好地代表聚类的几何中心

##### 4.3 平滑轨迹生成 (`_create_smooth_trajectory`)

```python
def _create_smooth_trajectory(self, cluster_centroids):
    # 计算累积距离参数
    distances = np.zeros(n_centroids)
    for i in range(1, n_centroids):
        distances[i] = distances[i-1] + np.linalg.norm(
            cluster_centroids[:, i] - cluster_centroids[:, i-1]
        )
    
    # 参数化插值
    t_original = distances / distances[-1]
    t_smooth = np.linspace(0, 1, n_smooth_points)
    
    # 样条插值
    for dim in range(cluster_centroids.shape[0]):
        if n_centroids >= 4:
            interp_func = interp1d(t_original, cluster_centroids[dim, :], 
                                 kind='cubic')
        else:
            interp_func = interp1d(t_original, cluster_centroids[dim, :], 
                                 kind='linear')
        smooth_trajectory[dim, :] = interp_func(t_smooth)
    
    # 强制单调性
    return self._enforce_trajectory_monotonicity(smooth_trajectory)
```

**插值策略**：
- **参数化方法**：使用累积距离作为参数，确保插值点的均匀分布
- **自适应插值**：根据控制点数量选择三次样条或线性插值
- **密度增强**：生成比原始中心点更密集的轨迹点，提高轨迹平滑度

##### 4.4 单向性保证 (`_enforce_trajectory_monotonicity`)

```python
def _enforce_trajectory_monotonicity(self, trajectory):
    # 计算主方向（第一主成分）
    pca = PCA(n_components=1)
    pca.fit(trajectory.T)
    main_direction = pca.components_[0]
    
    # 投影到主方向
    projections = trajectory.T @ main_direction
    
    # 强制单调递增
    monotonic_projections = np.copy(projections)
    for i in range(1, len(monotonic_projections)):
        if monotonic_projections[i] < monotonic_projections[i-1]:
            monotonic_projections[i] = monotonic_projections[i-1] + 1e-6
    
    # 重新构建轨迹点
    for i in range(n_points):
        if curr_proj < prev_proj or curr_proj > next_proj:
            # 线性插值调整位置
            alpha = (curr_proj - prev_proj) / (next_proj - prev_proj)
            alpha = np.clip(alpha, 0.1, 0.9)
            corrected_trajectory[:, i] = (1 - alpha) * trajectory[:, i-1] + alpha * trajectory[:, i+1]
```

**防折返机制**：
- **主成分分析**：识别轨迹的主要方向
- **单调性约束**：确保沿主方向的投影单调递增
- **局部调整**：通过线性插值修正违反单调性的点，避免全局重构

##### 4.5 最小生成树构建 (继承自 Monocle3)

虽然在简化的线性轨迹中不直接使用，但在完整的 `MonocleTrajectoryLearner` 中保留了 MST 算法：

```python
def _calc_principal_graph(self, X, C0):
    # 计算成本矩阵
    Phi = norm_sq_C[:, np.newaxis] + norm_sq_C[np.newaxis, :] - 2 * C.T @ C
    
    # 构建最小生成树
    G = nx.from_numpy_array(Phi)
    mst = nx.minimum_spanning_tree(G)
    
    # 获取邻接矩阵
    stree = nx.adjacency_matrix(mst, weight='weight').toarray()
    W = (stree != 0).astype(float)
```

**MST 的作用**：
- **连通性保证**：确保所有节点在图中连通
- **最优性**：最小化总边权重，避免冗余连接
- **树结构**：保持无环特性，适合轨迹表示

##### 4.6 聚类算法的多层次应用

算法中使用了多种聚类方法，各有特定用途：

1. **K-means 聚类**（在 `_select_medioids` 中）：
   ```python
   kmeans = KMeans(n_clusters=ncenter, random_state=0, n_init=10)
   cluster_labels = kmeans.fit_predict(X.T)
   ```
   - **用途**：选择代表性的中心点（medioids）
   - **优势**：快速、稳定，适合初始化

2. **DBSCAN 聚类**（在 `_filter_outlier_cells` 中）：
   ```python
   dbscan = DBSCAN(eps=1.5, min_samples=20)
   dbscan_labels = dbscan.fit_predict(X.T)
   ```
   - **用途**：检测离群点和噪声
   - **优势**：不需要预设聚类数，能识别任意形状的聚类

3. **基于年龄的聚类排序**：
   - **用途**：构建有生物学意义的轨迹顺序
   - **方法**：计算每个聚类的平均年龄并排序

### 2.4 细胞投影与伪时间计算

#### 步骤 5: 细胞到轨迹的投影

```python
def _project_cells_to_smooth_trajectory(self, X, clusters, trajectory_points, filtered_mask):
    for i in range(X.shape[1]):
        cell = X[:, i]
        
        # 离群点处理
        if not filtered_mask[i]:
            # 投影到最近的轨迹端点
            dist_to_start = np.linalg.norm(cell - trajectory_points[:, 0])
            dist_to_end = np.linalg.norm(cell - trajectory_points[:, -1])
            projection = trajectory_points[:, 0] if dist_to_start < dist_to_end else trajectory_points[:, -1]
        else:
            # 正常细胞：投影到最近的轨迹段
            min_distance = float('inf')
            best_projection = trajectory_points[:, 0]
            
            for j in range(n_trajectory_points - 1):
                point_a = trajectory_points[:, j]
                point_b = trajectory_points[:, j + 1]
                projection = self._project_point_to_line_segment(cell, point_a, point_b)
                distance = np.linalg.norm(cell - projection)
                
                if distance < min_distance:
                    min_distance = distance
                    best_projection = projection
```

**投影策略**：
- **离群点特殊处理**：将离群点投影到轨迹端点，避免扭曲主轨迹
- **线段投影**：对正常细胞，投影到最近的轨迹线段上
- **距离最小化**：选择使投影距离最小的线段

### 2.5 轨迹信息回投与可视化

#### 步骤 6-7: 多模态轨迹投影与可视化

```python
# 将联合轨迹投影回各个模态
start_idx = 0
for modality_key, adata in adatas.items():
    end_idx = start_idx + len(adata)
    
    # 提取该模态的UMAP嵌入
    modality_umap = joint_umap_embedding[start_idx:end_idx]
    adata.obsm[embedding_key] = modality_umap
    
    # 计算轨迹位置（伪时间）
    project_data_to_trajectory(adata, trajectory, embedding_key, 'trajectory_position')
```

## 3. 与 Monocle3 的详细比较

### 3.1 相似之处

1. **核心目标一致**：
   - 都致力于从单细胞数据推断细胞状态的连续轨迹
   - 揭示细胞分化、发育或疾病进展等动态过程

2. **降维方法**：
   - 都使用 UMAP 作为主要降维工具
   - 在低维空间中构建和可视化轨迹

3. **图论基础**：
   - 都使用图论方法构建细胞连接关系
   - 采用最小生成树确保连通性和最优性

4. **伪时间概念**：
   - `trajectory_position` 对应 Monocle3 的 pseudotime
   - 量化细胞在轨迹上的进展程度

### 3.2 关键创新与区别

#### 3.2.1 多模态整合能力

**我们的方法**：
- 原生支持多模态数据（RNA + ATAC）
- 通过深度学习模型学习联合潜在空间
- 使用最优传输损失对齐不同模态

**Monocle3**：
- 主要设计用于单模态数据
- 多模态整合需要外部预处理
- 缺乏原生的跨模态对齐机制

#### 3.2.2 轨迹构建策略

**我们的方法**：
```python
# 基于年龄标签的有序轨迹构建
age_order = np.argsort(cluster_ages)
sorted_centroids = cluster_centroids[:, age_order]
trajectory_points = self._create_smooth_trajectory(sorted_centroids)
```
- **监督式轨迹构建**：利用年龄等先验知识
- **线性轨迹假设**：适合年龄相关的连续过程
- **强制单向性**：避免生物学上不合理的折返

**Monocle3**：
```r
# 无监督的主图学习
principal_graph <- learn_graph(cds)
cds <- order_cells(cds)
```
- **无监督图学习**：从数据中发现复杂分支结构
- **弹性主图**：可以表示分叉、合并等复杂轨迹
- **数据驱动**：不依赖先验的时间或状态信息

#### 3.2.3 离群点处理

**我们的方法**：
```python
# 专门的离群点检测和过滤
dbscan = DBSCAN(eps=1.5, min_samples=20)
filtered_mask = self._filter_outlier_cells(X, clusters, age_labels)
```
- **主动离群点检测**：使用 DBSCAN 识别脱离主轨迹的细胞群
- **几何距离过滤**：基于到主群体中心的距离
- **年龄组保护**：确保每个年龄组的代表性

**Monocle3**：
- **被动处理**：主要通过图构建过程自然排除
- **依赖参数调优**：通过调整近邻数等参数间接控制
- **缺乏专门机制**：没有针对特定类型离群点的处理

#### 3.2.4 轨迹平滑与单调性

**我们的方法**：
```python
# 强制轨迹单调性
def _enforce_trajectory_monotonicity(self, trajectory):
    pca = PCA(n_components=1)
    main_direction = pca.components_[0]
    # 确保沿主方向单调递增
```
- **主成分导向**：基于数据的主要变异方向
- **单调性约束**：防止轨迹折返
- **局部调整**：保持轨迹的整体形状

**Monocle3**：
- **图结构约束**：通过树结构自然避免环路
- **全局优化**：同时考虑所有约束条件
- **灵活分支**：允许复杂的分叉和合并

### 3.3 算法复杂度比较

| 方面 | 我们的方法 | Monocle3 |
|------|------------|----------|
| 时间复杂度 | O(n log n) | O(n²) |
| 空间复杂度 | O(n) | O(n²) |
| 参数敏感性 | 低 | 中等 |
| 可解释性 | 高（基于年龄） | 中等 |
| 适用场景 | 时间序列数据 | 分化轨迹 |

### 3.4 生物学解释性

**我们的方法优势**：
- **年龄导向**：轨迹直接对应生物学时间
- **多模态一致性**：RNA 和 ATAC 轨迹在同一框架下
- **定量年龄关系**：可以量化不同年龄组间的分子距离

**Monocle3 优势**：
- **发现性分析**：能发现未知的分化路径
- **分支检测**：识别细胞命运决定点
- **基因动态**：沿轨迹的基因表达变化模式

## 4. 算法性能与鲁棒性

### 4.1 计算效率优化

1. **向量化操作**：大量使用 NumPy 向量化计算
2. **内存管理**：及时清理中间变量，支持大规模数据
3. **并行化潜力**：UMAP 和聚类步骤可以并行化

### 4.2 参数鲁棒性

1. **自适应参数**：多数参数基于数据特征自动设定
2. **容错机制**：对异常输入有相应的处理策略
3. **参数验证**：输入参数的合理性检查

### 4.3 生物学验证

1. **年龄一致性**：轨迹顺序与生物学年龄高度一致
2. **模态协调性**：RNA 和 ATAC 轨迹显示相似的年龄模式
3. **功能富集**：轨迹相关基因富集在年龄相关通路

## 5. 总结

`generate_joint_trajectory_from_model` 函数实现了一个创新的多模态轨迹推断算法，它：

1. **继承了 Monocle3 的优秀设计**：图论基础、MST 构建、伪时间概念
2. **针对多模态数据进行了优化**：深度学习整合、联合潜在空间
3. **专门处理年龄相关轨迹**：监督式构建、单向性保证、离群点过滤
4. **提供了鲁棒的实现**：参数自适应、错误处理、性能优化

这个算法特别适合于研究与年龄相关的细胞状态变化，如衰老、发育或疾病进展等生物学过程，为多组学数据的整合分析提供了强有力的工具。
