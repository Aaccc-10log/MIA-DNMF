from sklearn.cluster import KMeans

def k_means(H, n_clusters):
    """
    对节点嵌入 H 进行 K-Means 聚类
    
    参数:
        H: torch.Tensor, 形状为 (num_nodes, embedding_dim) 的节点嵌入
        n_clusters: int, 聚类数量
    
    返回:
        labels: torch.Tensor, 形状为 (num_nodes,), 包含每个节点的聚类标签
    """
    # 将 PyTorch 张量转换为 numpy 数组（sklearn 需要）
    H_np = H.detach().cpu().numpy() if H.is_cuda else H.numpy()
    
    # 执行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(H_np)
    
    return labels