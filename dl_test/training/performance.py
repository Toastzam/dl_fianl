import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix, top_k_accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import pandas as pd
import os
from model import SimCLRVIT
from dataset import StanfordDogsDataset, get_simclr_transforms
from torch.utils.data import DataLoader

# UMAP을 선택적으로 import (설치되어 있지 않으면 None)
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP이 설치되지 않았습니다. t-SNE와 PCA만 사용합니다.")

# GPU 메모리 최적화 설정
plt.style.use('default')
sns.set_palette("husl")

# 1. 모델 로드
print("1. 모델 로딩 중...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimCLRVIT(out_dim=128)
model.load_state_dict(torch.load(r'C:\dl_final\models\simclr_vit_dog_model_finetuned_v2.pth', map_location=device))
model.eval()
model.to(device)

# 2. 데이터셋 준비 (라벨 포함)
print("2. 데이터셋 준비 중...")
val_dataset = StanfordDogsDataset(root_dir=r'C:\dl_final\dl_test\training\Images', transform=get_simclr_transforms(224), sample_ratio=1.0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# 임베딩 추출
print("3. 임베딩 추출 중...")
all_embeds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (img1, img2, labels) in enumerate(val_loader):
        img1 = img1.to(device)
        embeds = model(img1).cpu().numpy()
        all_embeds.append(embeds)
        all_labels.append(labels.numpy())
        if batch_idx % 10 == 0:
            print(f"  배치 {batch_idx}/{len(val_loader)} 처리 완료")

all_embeds = np.concatenate(all_embeds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
num_classes = len(np.unique(all_labels))
print(f"총 {len(all_embeds)}개 샘플, {num_classes}개 클래스 처리 완료")

# 결과 저장을 위한 디렉토리 생성
os.makedirs('performance_plots', exist_ok=True)

# =====================
# 시각화 함수들
# =====================

def save_and_show_plot(filename):
    """플롯을 저장하고 표시하는 헬퍼 함수"""
    plt.savefig(f'performance_plots/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

def plot_embedding_analysis():
    """임베딩 차원 축소 및 클러스터링 분석"""
    print("\n=== 임베딩 차원 축소 및 클러스터링 분석 ===")
    
    # t-SNE 시각화
    print("t-SNE 계산 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeds_2d_tsne = tsne.fit_transform(all_embeds)
    
    # UMAP 시각화 (더 빠르고 구조 보존이 좋음)
    if UMAP_AVAILABLE:
        print("UMAP 계산 중...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embeds_2d_umap = umap_reducer.fit_transform(all_embeds)
    else:
        print("UMAP 미설치로 t-SNE 결과 재사용")
        embeds_2d_umap = embeds_2d_tsne
    
    # PCA 시각화
    print("PCA 계산 중...")
    pca = PCA(n_components=2)
    embeds_2d_pca = pca.fit_transform(all_embeds)
    
    # 서브플롯으로 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # t-SNE 원본
    scatter1 = axes[0,0].scatter(embeds_2d_tsne[:,0], embeds_2d_tsne[:,1], c=all_labels, cmap='tab20', alpha=0.7, s=10)
    axes[0,0].set_title('t-SNE (Ground Truth Labels)')
    
    # UMAP 원본
    scatter2 = axes[0,1].scatter(embeds_2d_umap[:,0], embeds_2d_umap[:,1], c=all_labels, cmap='tab20', alpha=0.7, s=10)
    axes[0,1].set_title('UMAP (Ground Truth Labels)')
    
    # PCA 원본
    scatter3 = axes[0,2].scatter(embeds_2d_pca[:,0], embeds_2d_pca[:,1], c=all_labels, cmap='tab20', alpha=0.7, s=10)
    axes[0,2].set_title(f'PCA (설명분산: {pca.explained_variance_ratio_.sum():.3f})')
    
    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeds)
    
    scatter4 = axes[1,0].scatter(embeds_2d_tsne[:,0], embeds_2d_tsne[:,1], c=cluster_labels, cmap='tab20', alpha=0.7, s=10)
    axes[1,0].set_title('t-SNE (KMeans Clustering)')
    
    scatter5 = axes[1,1].scatter(embeds_2d_umap[:,0], embeds_2d_umap[:,1], c=cluster_labels, cmap='tab20', alpha=0.7, s=10)
    axes[1,1].set_title('UMAP (KMeans Clustering)')
    
    # 클러스터링 품질 평가
    silhouette_avg = silhouette_score(all_embeds, cluster_labels)
    ari_score = adjusted_rand_score(all_labels, cluster_labels)
    
    axes[1,2].text(0.1, 0.7, f'클러스터링 품질 평가\n\nSilhouette Score: {silhouette_avg:.3f}\nAdjusted Rand Index: {ari_score:.3f}\n\n더 높은 값이 좋음', 
                   transform=axes[1,2].transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    axes[1,2].set_title('클러스터링 품질 평가')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    save_and_show_plot('embedding_analysis.png')
    
    return embeds_2d_tsne, embeds_2d_umap, embeds_2d_pca, cluster_labels

def plot_class_analysis():
    """클래스별 상세 분석"""
    print("\n=== 클래스별 상세 분석 ===")
    
    unique_labels = np.unique(all_labels)
    
    # 클래스별 centroid 계산 및 시각화
    centroids = np.array([all_embeds[all_labels==lbl].mean(axis=0) for lbl in unique_labels])
    
    # PCA로 centroid 시각화 (더 안정적)
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(centroids)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 클래스별 centroid 시각화
    scatter = axes[0,0].scatter(centroids_2d[:,0], centroids_2d[:,1], c=unique_labels, cmap='tab20', s=120, edgecolor='k')
    for i, lbl in enumerate(unique_labels):
        axes[0,0].text(centroids_2d[i,0], centroids_2d[i,1], str(lbl), fontsize=8)
    axes[0,0].set_title('Class Centroids (PCA)')
    
    # 임베딩 norm 분포 (클래스별)
    norms = np.linalg.norm(all_embeds, axis=1)
    data = [norms[all_labels==lbl] for lbl in unique_labels[:20]]  # 처음 20개 클래스만
    bp = axes[0,1].boxplot(data, labels=unique_labels[:20])
    axes[0,1].set_xlabel('Class Label')
    axes[0,1].set_ylabel('Embedding Norm')
    axes[0,1].set_title('Embedding Norm Distribution by Class (Top 20)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 클래스 간 거리 행렬
    centroid_distances = euclidean_distances(centroids)
    im = axes[1,0].imshow(centroid_distances, cmap='coolwarm')
    axes[1,0].set_title('Inter-class Distance Matrix')
    axes[1,0].set_xlabel('Class ID')
    axes[1,0].set_ylabel('Class ID')
    plt.colorbar(im, ax=axes[1,0])
    
    # 클래스 내 분산 vs 클래스 간 분산
    intra_class_var = [np.var(all_embeds[all_labels==lbl], axis=0).mean() for lbl in unique_labels]
    inter_class_var = np.var(centroids, axis=0).mean()
    
    axes[1,1].bar(range(len(intra_class_var[:20])), intra_class_var[:20])
    axes[1,1].axhline(y=inter_class_var, color='r', linestyle='--', label=f'Inter-class Var: {inter_class_var:.3f}')
    axes[1,1].set_xlabel('Class ID')
    axes[1,1].set_ylabel('Intra-class Variance')
    axes[1,1].set_title('Intra-class vs Inter-class Variance')
    axes[1,1].legend()
    
    plt.tight_layout()
    save_and_show_plot('class_analysis.png')

def plot_similarity_analysis():
    """유사도 분석 및 ROC"""
    print("\n=== 유사도 분석 ===")
    
    # 코사인 유사도 계산
    sim_matrix = cosine_similarity(all_embeds)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 유사도 행렬 히트맵 (샘플링)
    sample_idx = np.random.choice(len(all_embeds), size=min(200, len(all_embeds)), replace=False)
    sample_sim = sim_matrix[np.ix_(sample_idx, sample_idx)]
    sample_labels = all_labels[sample_idx]
    
    sns.heatmap(sample_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Cosine Similarity'}, ax=axes[0,0])
    axes[0,0].set_title(f'Cosine Similarity Matrix (Sample {len(sample_idx)})')
    
    # Positive/Negative 쌍 유사도 분포 (대각선 제외)
    pos_mask = all_labels[:,None] == all_labels[None,:]
    neg_mask = all_labels[:,None] != all_labels[None,:]
    
    # 대각선 제외 (자기 자신과의 유사도 제외)
    diag_mask = np.eye(len(all_embeds), dtype=bool)
    pos_mask = pos_mask & ~diag_mask
    neg_mask = neg_mask & ~diag_mask
    
    pos_sims = sim_matrix[pos_mask]
    neg_sims = sim_matrix[neg_mask]
    
    # 무한값 제거
    pos_sims = pos_sims[np.isfinite(pos_sims)]
    neg_sims = neg_sims[np.isfinite(neg_sims)]
    
    axes[0,1].hist(pos_sims, bins=50, alpha=0.6, label=f'Positive ({len(pos_sims)})', density=True)
    axes[0,1].hist(neg_sims, bins=50, alpha=0.6, label=f'Negative ({len(neg_sims)})', density=True)
    axes[0,1].set_xlabel('Cosine Similarity')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Similarity Distribution')
    axes[0,1].legend()
    
    # ROC Curve (대각선 제외)
    y_true = pos_mask.astype(int).flatten()
    y_score = sim_matrix.flatten()
    
    # 무한값 제거
    finite_mask = np.isfinite(y_score)
    y_true = y_true[finite_mask]
    y_score = y_score[finite_mask]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    axes[1,0].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', linewidth=2)
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curve (Pairwise Similarity)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 임베딩 차원별 분산
    embedding_var = np.var(all_embeds, axis=0)
    axes[1,1].plot(embedding_var, marker='o', markersize=2)
    axes[1,1].set_xlabel('Embedding Dimension')
    axes[1,1].set_ylabel('Variance')
    axes[1,1].set_title('Variance per Embedding Dimension')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show_plot('similarity_analysis.png')
    
    return pos_sims, neg_sims, roc_auc

def plot_nearest_neighbor_analysis():
    """최근접 이웃 기반 분석"""
    print("\n=== 최근접 이웃 분석 ===")
    
    # 최근접 이웃 검색
    knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(all_embeds)
    distances, indices = knn.kneighbors(all_embeds)
    
    # Top-1, Top-5 정확도
    pred_labels = all_labels[indices[:,1]]  # 첫 번째는 자기 자신
    top1_acc = np.mean(all_labels == pred_labels)
    
    top5_hits = [all_labels[i] in all_labels[indices[i,1:6]] for i in range(len(all_labels))]
    top5_acc = np.mean(top5_hits)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, pred_labels)
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', ax=axes[0,0])
    axes[0,0].set_xlabel('Predicted Label (NN)')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_title(f'Confusion Matrix (Top-1 Acc: {top1_acc:.3f})')
    
    # 최근접 이웃 거리 분포
    nn_distances = distances[:,1]  # 첫 번째 최근접 이웃 거리
    correct_mask = all_labels == pred_labels
    
    axes[0,1].hist(nn_distances[correct_mask], bins=30, alpha=0.6, label='Correct', density=True)
    axes[0,1].hist(nn_distances[~correct_mask], bins=30, alpha=0.6, label='Incorrect', density=True)
    axes[0,1].set_xlabel('Cosine Distance to Nearest Neighbor')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('NN Distance Distribution')
    axes[0,1].legend()
    
    # Top-K 정확도 곡선
    k_values = range(1, min(21, len(np.unique(all_labels))))
    topk_accs = []
    for k in k_values:
        topk_hits = [all_labels[i] in all_labels[indices[i,1:k+1]] for i in range(len(all_labels))]
        topk_accs.append(np.mean(topk_hits))
    
    axes[1,0].plot(k_values, topk_accs, marker='o', linewidth=2)
    axes[1,0].set_xlabel('K (Top-K)')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('Top-K Accuracy Curve')
    axes[1,0].grid(True, alpha=0.3)
    
    # 정확도 요약
    summary_text = f"""
    성능 요약:
    
    Top-1 Accuracy: {top1_acc:.3f}
    Top-5 Accuracy: {top5_acc:.3f}
    
    평균 NN 거리:
    - 정답: {nn_distances[correct_mask].mean():.3f}
    - 오답: {nn_distances[~correct_mask].mean():.3f}
    
    총 샘플 수: {len(all_embeds)}
    클래스 수: {num_classes}
    """
    
    axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))
    axes[1,1].set_title('성능 요약')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    save_and_show_plot('nearest_neighbor_analysis.png')
    
    return top1_acc, top5_acc

def plot_learning_curves():
    """학습 곡선 시각화"""
    print("\n=== 학습 곡선 분석 ===")
    
    def load_and_plot_csv(csv_path, title_prefix, ax_loss, ax_acc):
        if not os.path.exists(csv_path):
            print(f'{csv_path} 파일을 찾을 수 없습니다.')
            return
        
        df = pd.read_csv(csv_path)
        
        ax_loss.plot(df['epoch'], df['train_loss'], label=f'{title_prefix} Train Loss', linewidth=2)
        ax_loss.plot(df['epoch'], df['val_loss'], label=f'{title_prefix} Val Loss', linewidth=2)
        
        ax_acc.plot(df['epoch'], df['train_acc'], label=f'{title_prefix} Train Acc', linewidth=2)
        ax_acc.plot(df['epoch'], df['val_acc'], label=f'{title_prefix} Val Acc', linewidth=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 곡선
    load_and_plot_csv('linear_eval_log.csv', 'Linear Eval', axes[0], axes[1])
    load_and_plot_csv('semi_eval_log.csv', 'Semi-supervised', axes[0], axes[1])
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show_plot('learning_curves.png')

def plot_advanced_visualizations():
    """고급 시각화 함수들"""
    print("\n=== 고급 시각화 분석 ===")
    
    # 1. 임베딩 품질 분석: 동일 클래스 vs 다른 클래스 거리 분포
    print("임베딩 품질 분석 중...")
    same_class_distances = []
    diff_class_distances = []
    
    # 샘플링으로 계산 부하 줄이기
    sample_size = min(1000, len(all_embeds))
    sample_indices = np.random.choice(len(all_embeds), sample_size, replace=False)
    
    for i in sample_indices[:100]:  # 처음 100개만 계산
        for j in sample_indices:
            if i != j:
                dist = np.linalg.norm(all_embeds[i] - all_embeds[j])
                if all_labels[i] == all_labels[j]:
                    same_class_distances.append(dist)
                else:
                    diff_class_distances.append(dist)
    
    # 2. 클래스별 분리도 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 거리 분포
    if len(same_class_distances) > 0 and len(diff_class_distances) > 0:
        axes[0,0].hist(same_class_distances, bins=50, alpha=0.6, label='Same Class', density=True)
        axes[0,0].hist(diff_class_distances, bins=50, alpha=0.6, label='Different Class', density=True)
        axes[0,0].set_xlabel('Euclidean Distance')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Intra-class vs Inter-class Distance Distribution')
        axes[0,0].legend()
    else:
        axes[0,0].text(0.5, 0.5, 'Insufficient data for distance analysis', ha='center', va='center')
        axes[0,0].set_title('Distance Distribution (No Data)')
    
    # 3. 임베딩 차원별 중요도 (분산 기반)
    embedding_var = np.var(all_embeds, axis=0)
    top_dims = np.argsort(embedding_var)[-20:]  # 상위 20개 차원
    
    axes[0,1].bar(range(len(top_dims)), embedding_var[top_dims])
    axes[0,1].set_xlabel('Top Embedding Dimensions')
    axes[0,1].set_ylabel('Variance')
    axes[0,1].set_title('Top 20 Most Variant Embedding Dimensions')
    
    # 4. 클래스별 샘플 수 분포
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    axes[0,2].bar(range(len(counts)), counts)
    axes[0,2].set_xlabel('Class ID')
    axes[0,2].set_ylabel('Sample Count')
    axes[0,2].set_title('Samples per Class')
    
    # 5. 임베딩 norm 히스토그램
    norms = np.linalg.norm(all_embeds, axis=1)
    axes[1,0].hist(norms, bins=50, alpha=0.7)
    axes[1,0].axvline(norms.mean(), color='r', linestyle='--', label=f'Mean: {norms.mean():.2f}')
    axes[1,0].set_xlabel('Embedding Norm')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Embedding Norm Distribution')
    axes[1,0].legend()
    
    # 6. 클래스별 평균 유사도
    sim_matrix = cosine_similarity(all_embeds)
    class_avg_sim = []
    for label in unique_labels:
        mask = all_labels == label
        if np.sum(mask) > 1:
            class_sim = sim_matrix[mask][:, mask]
            np.fill_diagonal(class_sim, np.nan)
            avg_sim = np.nanmean(class_sim)
            class_avg_sim.append(avg_sim)
        else:
            class_avg_sim.append(0)
    
    axes[1,1].bar(range(len(class_avg_sim)), class_avg_sim)
    axes[1,1].set_xlabel('Class ID')
    axes[1,1].set_ylabel('Average Intra-class Similarity')
    axes[1,1].set_title('Average Similarity within Each Class')
    
    # 7. 차원별 PCA 설명 분산
    pca_full = PCA()
    pca_full.fit(all_embeds)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    axes[1,2].plot(range(1, len(cumsum_var)+1), cumsum_var, marker='o', markersize=3)
    axes[1,2].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    axes[1,2].set_xlabel('Number of Components')
    axes[1,2].set_ylabel('Cumulative Explained Variance')
    axes[1,2].set_title('PCA Explained Variance')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show_plot('advanced_analysis.png')
    
    return same_class_distances, diff_class_distances

def plot_data_augmentation_effect():
    """데이터 증강 효과 시각화"""
    print("\n=== 데이터 증강 효과 분석 ===")
    
    # 원본 이미지와 증강된 이미지 비교
    sample_indices = np.random.choice(len(val_dataset), size=6, replace=False)
    
    fig, axes = plt.subplots(6, 3, figsize=(12, 18))
    
    for i, idx in enumerate(sample_indices):
        # 원본 이미지
        img1, img2, label = val_dataset[idx]
        
        # 이미지를 표시 가능한 형태로 변환
        def tensor_to_image(tensor):
            # 정규화 해제 (ImageNet 표준)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img = tensor.clone()
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            
            img = torch.clamp(img, 0, 1)
            return np.transpose(img.numpy(), (1, 2, 0))
        
        # 원본 및 증강된 이미지 표시
        try:
            img1_display = tensor_to_image(img1)
            img2_display = tensor_to_image(img2)
            
            axes[i,0].imshow(img1_display)
            axes[i,0].set_title(f'Original (Class {label})')
            axes[i,0].axis('off')
            
            axes[i,1].imshow(img2_display)
            axes[i,1].set_title(f'Augmented (Class {label})')
            axes[i,1].axis('off')
            
            # 두 이미지의 픽셀 차이
            diff = np.abs(img1_display - img2_display)
            im = axes[i,2].imshow(diff, cmap='hot')
            axes[i,2].set_title('Pixel Difference')
            axes[i,2].axis('off')
            
        except Exception as e:
            # 표시 실패시 빈 플롯
            axes[i,0].text(0.5, 0.5, f'Display Error\n{str(e)[:50]}', ha='center', va='center')
            axes[i,0].axis('off')
            axes[i,1].axis('off')
            axes[i,2].axis('off')
    
    plt.suptitle('Data Augmentation Effect Comparison', fontsize=16)
    plt.tight_layout()
    save_and_show_plot('augmentation_effect.png')

def plot_embedding_evolution():
    """임베딩 진화 분석 (서로 다른 클래스 간 관계)"""
    print("\n=== 임베딩 공간 구조 분석 ===")
    
    # 클래스 간 관계 분석
    unique_labels = np.unique(all_labels)
    
    # 클래스별 centroid 계산
    centroids = np.array([all_embeds[all_labels==lbl].mean(axis=0) for lbl in unique_labels])
    
    # 클래스 간 유사도 행렬
    centroid_sim = cosine_similarity(centroids)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 클래스 간 유사도 히트맵
    im1 = axes[0,0].imshow(centroid_sim, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0,0].set_title('Inter-class Centroid Similarity Matrix')
    axes[0,0].set_xlabel('Class ID')
    axes[0,0].set_ylabel('Class ID')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 가장 유사한/다른 클래스 쌍 찾기
    np.fill_diagonal(centroid_sim, -2)  # 자기 자신 제외
    most_similar_idx = np.unravel_index(np.argmax(centroid_sim), centroid_sim.shape)
    most_different_idx = np.unravel_index(np.argmin(centroid_sim), centroid_sim.shape)
    
    # 클래스별 분산 vs 다른 클래스와의 거리
    intra_class_var = []
    inter_class_dist = []
    
    for i, label in enumerate(unique_labels):
        # 클래스 내 분산
        class_embeds = all_embeds[all_labels == label]
        if len(class_embeds) > 1:
            var = np.var(class_embeds, axis=0).mean()
            intra_class_var.append(var)
            
            # 다른 클래스 centroid와의 평균 거리
            other_centroids = centroids[unique_labels != label]
            if len(other_centroids) > 0:
                avg_dist = np.mean([np.linalg.norm(centroids[i] - c) for c in other_centroids])
                inter_class_dist.append(avg_dist)
            else:
                inter_class_dist.append(0)
        else:
            intra_class_var.append(0)
            inter_class_dist.append(0)
    
    # 분산 vs 거리 산점도
    axes[0,1].scatter(intra_class_var, inter_class_dist, alpha=0.6)
    axes[0,1].set_xlabel('Intra-class Variance')
    axes[0,1].set_ylabel('Inter-class Distance')
    axes[0,1].set_title('Class Separation Quality')
    axes[0,1].grid(True, alpha=0.3)
    
    # 임베딩 공간의 밀도 분포 (2D 히스토그램)
    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(all_embeds)
    
    h = axes[1,0].hist2d(embeds_2d[:,0], embeds_2d[:,1], bins=50, cmap='Blues')
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[1,0].set_title('Embedding Density Distribution (PCA)')
    plt.colorbar(h[3], ax=axes[1,0])
    
    # 클러스터 분리도 품질 요약
    silhouette_avg = silhouette_score(all_embeds, all_labels)
    
    summary_text = f"""
    임베딩 공간 품질 요약:
    
    Silhouette Score: {silhouette_avg:.3f}
    
    가장 유사한 클래스:
    {unique_labels[most_similar_idx[0]]} - {unique_labels[most_similar_idx[1]]}
    (유사도: {centroid_sim[most_similar_idx]:.3f})
    
    가장 다른 클래스:
    {unique_labels[most_different_idx[0]]} - {unique_labels[most_different_idx[1]]}
    (유사도: {centroid_sim[most_different_idx]:.3f})
    
    평균 클래스 내 분산: {np.mean(intra_class_var):.3f}
    평균 클래스 간 거리: {np.mean(inter_class_dist):.3f}
    """
    
    axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.7))
    axes[1,1].set_title('임베딩 공간 품질 요약')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    save_and_show_plot('embedding_structure.png')

def plot_model_interpretation():
    """모델 해석 및 특성 분석"""
    print("\n=== 모델 해석 및 특성 분석 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 임베딩 활성화 패턴 히트맵
    print("임베딩 활성화 패턴 분석 중...")
    # 클래스별 평균 임베딩 계산
    unique_labels = np.unique(all_labels)
    class_embeddings = []
    for label in unique_labels[:10]:  # 처음 10개 클래스만
        class_mask = all_labels == label
        if np.sum(class_mask) > 0:
            avg_embedding = all_embeds[class_mask].mean(axis=0)
            class_embeddings.append(avg_embedding)
    
    class_embeddings = np.array(class_embeddings)
    im1 = axes[0,0].imshow(class_embeddings.T, cmap='viridis', aspect='auto')
    axes[0,0].set_title('Class-wise Average Embedding Patterns')
    axes[0,0].set_xlabel('Class ID')
    axes[0,0].set_ylabel('Embedding Dimension')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. 임베딩 차원 상관관계 분석
    print("차원 간 상관관계 분석 중...")
    # 랜덤 샘플링으로 계산 부하 줄이기
    sample_embeds = all_embeds[np.random.choice(len(all_embeds), min(1000, len(all_embeds)), replace=False)]
    correlation_matrix = np.corrcoef(sample_embeds.T)
    
    im2 = axes[0,1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0,1].set_title('Embedding Dimension Correlation')
    axes[0,1].set_xlabel('Dimension')
    axes[0,1].set_ylabel('Dimension')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. 클래스 분리 품질 분석 (Silhouette Analysis)
    print("Silhouette 분석 중...")
    from sklearn.metrics import silhouette_samples
    
    # 샘플링으로 계산 부하 줄이기
    sample_indices = np.random.choice(len(all_embeds), min(1000, len(all_embeds)), replace=False)
    sample_embeds = all_embeds[sample_indices]
    sample_labels = all_labels[sample_indices]
    
    silhouette_scores = silhouette_samples(sample_embeds, sample_labels)
    
    # 클래스별 Silhouette score 분포
    y_lower = 10
    for i, label in enumerate(np.unique(sample_labels)[:15]):  # 처음 15개 클래스만
        cluster_silhouette_values = silhouette_scores[sample_labels == label]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(sample_labels)))
        axes[0,2].fill_betweenx(np.arange(y_lower, y_upper),
                               0, cluster_silhouette_values,
                               facecolor=color, edgecolor=color, alpha=0.7)
        
        axes[0,2].text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10
    
    axes[0,2].set_xlabel('Silhouette Coefficient Values')
    axes[0,2].set_ylabel('Cluster Label')
    axes[0,2].set_title('Silhouette Analysis by Class')
    
    # 4. 임베딩 거리 분포 (L2 vs Cosine)
    print("다양한 거리 메트릭 비교 중...")
    sample_size = 500
    sample_idx = np.random.choice(len(all_embeds), sample_size, replace=False)
    sample_embeds = all_embeds[sample_idx]
    
    # L2 거리
    from sklearn.metrics.pairwise import euclidean_distances
    l2_distances = euclidean_distances(sample_embeds).flatten()
    l2_distances = l2_distances[l2_distances > 0]  # 자기 자신과의 거리 제외
    
    # 코사인 거리
    cosine_distances = 1 - cosine_similarity(sample_embeds)
    cosine_distances = cosine_distances.flatten()
    cosine_distances = cosine_distances[cosine_distances > 0]
    
    axes[1,0].hist(l2_distances, bins=50, alpha=0.6, label='L2 Distance', density=True)
    axes[1,0].hist(cosine_distances, bins=50, alpha=0.6, label='Cosine Distance', density=True)
    axes[1,0].set_xlabel('Distance')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distance Distribution Comparison')
    axes[1,0].legend()
    
    # 5. 클래스별 임베딩 분산 분석
    print("클래스별 분산 분석 중...")
    class_variances = []
    class_stds = []
    for label in unique_labels:
        class_embeds = all_embeds[all_labels == label]
        if len(class_embeds) > 1:
            var = np.var(class_embeds, axis=0).mean()
            std = np.std(class_embeds, axis=0).mean()
            class_variances.append(var)
            class_stds.append(std)
        else:
            class_variances.append(0)
            class_stds.append(0)
    
    axes[1,1].scatter(range(len(class_variances)), class_variances, alpha=0.6, label='Variance')
    axes[1,1].scatter(range(len(class_stds)), class_stds, alpha=0.6, label='Std Dev')
    axes[1,1].set_xlabel('Class ID')
    axes[1,1].set_ylabel('Spread Measure')
    axes[1,1].set_title('Class-wise Embedding Spread')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. 임베딩 품질 요약 통계
    embedding_stats = {
        'Mean Norm': np.linalg.norm(all_embeds, axis=1).mean(),
        'Std Norm': np.linalg.norm(all_embeds, axis=1).std(),
        'Mean Cosine Sim': np.mean(cosine_similarity(all_embeds)),
        'Embedding Sparsity': np.mean(np.abs(all_embeds) < 0.01),
        'Max Activation': np.max(all_embeds),
        'Min Activation': np.min(all_embeds)
    }
    
    stats_text = "임베딩 품질 통계:\n\n"
    for key, value in embedding_stats.items():
        stats_text += f"{key}: {value:.4f}\n"
    
    axes[1,2].text(0.1, 0.5, stats_text, transform=axes[1,2].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor='lightcyan', alpha=0.8))
    axes[1,2].set_title('임베딩 품질 통계')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    save_and_show_plot('model_interpretation.png')

def plot_clustering_comparison():
    """다양한 클러스터링 알고리즘 비교"""
    print("\n=== 클러스터링 알고리즘 비교 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA로 2D 변환 (시각화용)
    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(all_embeds)
    
    # 1. K-Means
    print("K-Means 클러스터링...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    kmeans_labels = kmeans.fit_predict(all_embeds)
    
    scatter1 = axes[0,0].scatter(embeds_2d[:,0], embeds_2d[:,1], c=kmeans_labels, cmap='tab20', alpha=0.6, s=10)
    axes[0,0].set_title(f'K-Means (ARI: {adjusted_rand_score(all_labels, kmeans_labels):.3f})')
    
    # 2. DBSCAN
    print("DBSCAN 클러스터링...")
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(all_embeds)
    
    scatter2 = axes[0,1].scatter(embeds_2d[:,0], embeds_2d[:,1], c=dbscan_labels, cmap='tab20', alpha=0.6, s=10)
    axes[0,1].set_title(f'DBSCAN (Clusters: {len(np.unique(dbscan_labels[dbscan_labels != -1]))})')
    
    # 3. Agglomerative Clustering
    print("Agglomerative 클러스터링...")
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=num_classes)
    agg_labels = agg.fit_predict(all_embeds)
    
    scatter3 = axes[0,2].scatter(embeds_2d[:,0], embeds_2d[:,1], c=agg_labels, cmap='tab20', alpha=0.6, s=10)
    axes[0,2].set_title(f'Agglomerative (ARI: {adjusted_rand_score(all_labels, agg_labels):.3f})')
    
    # 4. Ground Truth
    scatter4 = axes[1,0].scatter(embeds_2d[:,0], embeds_2d[:,1], c=all_labels, cmap='tab20', alpha=0.6, s=10)
    axes[1,0].set_title('Ground Truth Labels')
    
    # 5. 클러스터링 성능 비교
    clustering_scores = {
        'K-Means': {
            'ARI': adjusted_rand_score(all_labels, kmeans_labels),
            'Silhouette': silhouette_score(all_embeds, kmeans_labels)
        },
        'Agglomerative': {
            'ARI': adjusted_rand_score(all_labels, agg_labels),
            'Silhouette': silhouette_score(all_embeds, agg_labels)
        }
    }
    
    methods = list(clustering_scores.keys())
    ari_scores = [clustering_scores[method]['ARI'] for method in methods]
    silhouette_scores = [clustering_scores[method]['Silhouette'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1,1].bar(x - width/2, ari_scores, width, label='ARI', alpha=0.7)
    axes[1,1].bar(x + width/2, silhouette_scores, width, label='Silhouette', alpha=0.7)
    axes[1,1].set_xlabel('Clustering Method')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Clustering Performance Comparison')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(methods)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. 클러스터 크기 분포
    cluster_sizes = {
        'Ground Truth': np.bincount(all_labels),
        'K-Means': np.bincount(kmeans_labels),
        'Agglomerative': np.bincount(agg_labels)
    }
    
    for i, (method, sizes) in enumerate(cluster_sizes.items()):
        axes[1,2].hist(sizes, bins=20, alpha=0.6, label=method, density=True)
    
    axes[1,2].set_xlabel('Cluster Size')
    axes[1,2].set_ylabel('Density')
    axes[1,2].set_title('Cluster Size Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show_plot('clustering_comparison.png')

def plot_temporal_analysis():
    """학습 과정 시간적 분석"""
    print("\n=== 학습 과정 시간적 분석 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 학습 곡선의 변화율 분석
    def analyze_learning_curve(csv_path, title, ax):
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{csv_path} not found', ha='center', va='center')
            ax.set_title(f'{title} (No Data)')
            return
        
        df = pd.read_csv(csv_path)
        
        # 변화율 계산
        train_loss_diff = np.diff(df['train_loss'])
        val_loss_diff = np.diff(df['val_loss'])
        train_acc_diff = np.diff(df['train_acc'])
        val_acc_diff = np.diff(df['val_acc'])
        
        epochs = df['epoch'][1:]
        
        ax.plot(epochs, train_loss_diff, label='Train Loss Change', alpha=0.7)
        ax.plot(epochs, val_loss_diff, label='Val Loss Change', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Change Rate')
        ax.set_title(f'{title} - Loss Change Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    analyze_learning_curve('linear_eval_log.csv', 'Linear Evaluation', axes[0,0])
    analyze_learning_curve('semi_eval_log.csv', 'Semi-supervised', axes[0,1])
    
    # 2. 수렴 분석
    def plot_convergence_analysis(csv_path, title, ax):
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{csv_path} not found', ha='center', va='center')
            ax.set_title(f'{title} Convergence (No Data)')
            return
        
        df = pd.read_csv(csv_path)
        
        # 이동평균 계산
        window = 3
        if len(df) >= window:
            val_loss_smooth = df['val_loss'].rolling(window=window).mean()
            val_acc_smooth = df['val_acc'].rolling(window=window).mean()
            
            ax.plot(df['epoch'], df['val_loss'], alpha=0.3, label='Val Loss (Raw)')
            ax.plot(df['epoch'], val_loss_smooth, label='Val Loss (Smooth)', linewidth=2)
            
            ax2 = ax.twinx()
            ax2.plot(df['epoch'], df['val_acc'], alpha=0.3, color='orange', label='Val Acc (Raw)')
            ax2.plot(df['epoch'], val_acc_smooth, color='red', label='Val Acc (Smooth)', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss', color='blue')
            ax2.set_ylabel('Accuracy', color='red')
            ax.set_title(f'{title} - Convergence Analysis')
            ax.grid(True, alpha=0.3)
    
    plot_convergence_analysis('linear_eval_log.csv', 'Linear Evaluation', axes[0,2])
    
    # 3. 성능 개선 패턴
    def plot_improvement_pattern(csv_path, title, ax):
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{csv_path} not found', ha='center', va='center')
            ax.set_title(f'{title} Improvement (No Data)')
            return
        
        df = pd.read_csv(csv_path)
        
        # 최고 성능 업데이트 지점 찾기
        best_val_acc = df['val_acc'].cummax()
        improvement_points = (df['val_acc'] == best_val_acc) & (df['val_acc'] > df['val_acc'].shift(1, fill_value=0))
        
        ax.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', alpha=0.7)
        ax.plot(df['epoch'], best_val_acc, label='Best So Far', linewidth=2)
        ax.scatter(df['epoch'][improvement_points], df['val_acc'][improvement_points], 
                  color='red', s=50, label='Improvements', zorder=5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'{title} - Performance Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plot_improvement_pattern('linear_eval_log.csv', 'Linear Evaluation', axes[1,0])
    plot_improvement_pattern('semi_eval_log.csv', 'Semi-supervised', axes[1,1])
    
    # 4. 학습 안정성 분석
    def plot_stability_analysis(csv_path, title, ax):
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{csv_path} not found', ha='center', va='center')
            ax.set_title(f'{title} Stability (No Data)')
            return
        
        df = pd.read_csv(csv_path)
        
        # 분산 계산 (rolling window)
        window = 3
        if len(df) >= window:
            val_loss_var = df['val_loss'].rolling(window=window).var()
            val_acc_var = df['val_acc'].rolling(window=window).var()
            
            ax.plot(df['epoch'], val_loss_var, label='Val Loss Variance', alpha=0.7)
            ax.plot(df['epoch'], val_acc_var, label='Val Acc Variance', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Variance')
            ax.set_title(f'{title} - Training Stability')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plot_stability_analysis('linear_eval_log.csv', 'Linear Evaluation', axes[1,2])
    
    plt.tight_layout()
    save_and_show_plot('temporal_analysis.png')

def plot_error_analysis():
    """오류 분석 및 실패 케이스 분석"""
    print("\n=== 오류 분석 ===")
    
    # 최근접 이웃으로 예측
    knn = NearestNeighbors(n_neighbors=2, metric='cosine').fit(all_embeds)
    distances, indices = knn.kneighbors(all_embeds)
    pred_labels = all_labels[indices[:,1]]  # 자기 자신 제외한 가장 가까운 이웃
    
    # 정답/오답 분석
    correct_mask = all_labels == pred_labels
    incorrect_mask = ~correct_mask
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 클래스별 정확도
    unique_labels = np.unique(all_labels)
    class_accuracies = []
    for label in unique_labels:
        class_mask = all_labels == label
        if np.sum(class_mask) > 0:
            class_acc = np.mean(correct_mask[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    axes[0,0].bar(range(len(class_accuracies)), class_accuracies, alpha=0.7)
    axes[0,0].set_xlabel('Class ID')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('Per-Class Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 오류 유형 분석 (Confusion Matrix)
    cm = confusion_matrix(all_labels, pred_labels)
    
    # 가장 자주 혼동되는 클래스 쌍 찾기
    np.fill_diagonal(cm, 0)  # 대각선 제거
    top_confusions = np.unravel_index(np.argsort(cm.ravel())[-10:], cm.shape)
    
    im = axes[0,1].imshow(cm, cmap='Blues')
    axes[0,1].set_title('Confusion Matrix')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('True')
    plt.colorbar(im, ax=axes[0,1])
    
    # 3. 거리별 정확도
    nn_distances = distances[:,1]
    
    # 거리를 구간별로 나누어 정확도 계산
    distance_bins = np.linspace(nn_distances.min(), nn_distances.max(), 10)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(distance_bins)-1):
        bin_mask = (nn_distances >= distance_bins[i]) & (nn_distances < distance_bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(correct_mask[bin_mask])
            bin_accuracies.append(bin_acc)
            bin_centers.append((distance_bins[i] + distance_bins[i+1]) / 2)
    
    axes[0,2].plot(bin_centers, bin_accuracies, marker='o', linewidth=2)
    axes[0,2].set_xlabel('Nearest Neighbor Distance')
    axes[0,2].set_ylabel('Accuracy')
    axes[0,2].set_title('Accuracy vs NN Distance')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 오답 케이스의 임베딩 분포
    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(all_embeds)
    
    axes[1,0].scatter(embeds_2d[correct_mask,0], embeds_2d[correct_mask,1], 
                     alpha=0.6, s=10, label='Correct', color='green')
    axes[1,0].scatter(embeds_2d[incorrect_mask,0], embeds_2d[incorrect_mask,1], 
                     alpha=0.6, s=10, label='Incorrect', color='red')
    axes[1,0].set_xlabel('PC1')
    axes[1,0].set_ylabel('PC2')
    axes[1,0].set_title('Correct vs Incorrect Predictions')
    axes[1,0].legend()
    
    # 5. 임베딩 품질과 정확도의 관계
    embedding_norms = np.linalg.norm(all_embeds, axis=1)
    
    # Norm을 구간별로 나누어 정확도 계산
    norm_bins = np.linspace(embedding_norms.min(), embedding_norms.max(), 10)
    norm_accuracies = []
    norm_centers = []
    
    for i in range(len(norm_bins)-1):
        bin_mask = (embedding_norms >= norm_bins[i]) & (embedding_norms < norm_bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(correct_mask[bin_mask])
            norm_accuracies.append(bin_acc)
            norm_centers.append((norm_bins[i] + norm_bins[i+1]) / 2)
    
    axes[1,1].plot(norm_centers, norm_accuracies, marker='o', linewidth=2)
    axes[1,1].set_xlabel('Embedding Norm')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('Accuracy vs Embedding Norm')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. 오류 통계 요약
    error_stats = {
        'Total Samples': len(all_labels),
        'Correct Predictions': np.sum(correct_mask),
        'Incorrect Predictions': np.sum(incorrect_mask),
        'Overall Accuracy': np.mean(correct_mask),
        'Worst Class Acc': np.min(class_accuracies),
        'Best Class Acc': np.max(class_accuracies),
        'Avg NN Distance (Correct)': nn_distances[correct_mask].mean(),
        'Avg NN Distance (Incorrect)': nn_distances[incorrect_mask].mean()
    }
    
    stats_text = "오류 분석 통계:\n\n"
    for key, value in error_stats.items():
        if isinstance(value, float):
            stats_text += f"{key}: {value:.4f}\n"
        else:
            stats_text += f"{key}: {value}\n"
    
    axes[1,2].text(0.1, 0.5, stats_text, transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
    axes[1,2].set_title('오류 분석 통계')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    save_and_show_plot('error_analysis.png')

# =====================
# 메인 실행 부분
# =====================

if __name__ == "__main__":
    # 모든 시각화 실행
    print("="*60)
    print("SimCLR 모델 성능 종합 분석 시작")
    print("="*60)
    
    embeds_2d_tsne, embeds_2d_umap, embeds_2d_pca, cluster_labels = plot_embedding_analysis()
    plot_class_analysis()
    pos_sims, neg_sims, roc_auc = plot_similarity_analysis()
    top1_acc, top5_acc = plot_nearest_neighbor_analysis()
    plot_learning_curves()
    
    # 새로운 고급 시각화들
    same_class_distances, diff_class_distances = plot_advanced_visualizations()
    plot_data_augmentation_effect()
    plot_embedding_evolution()
    plot_model_interpretation()
    plot_clustering_comparison()
    plot_temporal_analysis()
    plot_error_analysis()
    
    # 최종 성능 요약
    print("\n" + "="*60)
    print("최종 성능 요약")
    print("="*60)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Positive Similarity 평균: {pos_sims.mean():.4f}")
    print(f"Negative Similarity 평균: {neg_sims.mean():.4f}")
    
    # 거리 통계 (데이터가 있는 경우에만)
    if len(same_class_distances) > 0:
        print(f"Same Class Distance 평균: {np.mean(same_class_distances):.4f}")
    else:
        print("Same Class Distance: 데이터 부족")
        
    if len(diff_class_distances) > 0:
        print(f"Different Class Distance 평균: {np.mean(diff_class_distances):.4f}")
    else:
        print("Different Class Distance: 데이터 부족")
    
    print(f"총 처리된 샘플 수: {len(all_embeds)}")
    print(f"클래스 수: {num_classes}")
    print(f"임베딩 차원: {all_embeds.shape[1]}")
    print("="*60)
    
    # 생성된 시각화 파일 목록
    plot_files = [
        'embedding_analysis.png',          # t-SNE, UMAP, PCA 비교
        'class_analysis.png',              # 클래스별 상세 분석
        'similarity_analysis.png',         # 유사도 분석 및 ROC
        'nearest_neighbor_analysis.png',   # 최근접 이웃 기반 분석
        'learning_curves.png',             # 학습 곡선
        'advanced_analysis.png',           # 고급 통계 분석
        'augmentation_effect.png',         # 데이터 증강 효과
        'embedding_structure.png',         # 임베딩 공간 구조
        'model_interpretation.png',        # 모델 해석 및 특성 분석
        'clustering_comparison.png',       # 클러스터링 알고리즘 비교
        'temporal_analysis.png',           # 학습 과정 시간적 분석
        'error_analysis.png'               # 오류 분석 및 실패 케이스
    ]
    
    print(f"\n생성된 시각화 파일들 ({len(plot_files)}개):")
    descriptions = {
        'embedding_analysis.png': 'Embedding 차원축소 비교 (t-SNE, UMAP, PCA)',
        'class_analysis.png': '클래스별 상세 분석 (centroid, norm, 거리)',
        'similarity_analysis.png': '유사도 분석 (ROC, 분포, 차원별 분산)',
        'nearest_neighbor_analysis.png': '최근접 이웃 분석 (Top-K, confusion matrix)',
        'learning_curves.png': '학습 곡선 (loss, accuracy)',
        'advanced_analysis.png': '고급 통계 분석 (거리분포, 분산, PCA)',
        'augmentation_effect.png': '데이터 증강 효과 비교',
        'embedding_structure.png': '임베딩 공간 구조 분석',
        'model_interpretation.png': '모델 해석 (활성화패턴, 상관관계, Silhouette)',
        'clustering_comparison.png': '클러스터링 알고리즘 비교 (K-means, DBSCAN 등)',
        'temporal_analysis.png': '학습과정 시간적 분석 (수렴, 안정성)',
        'error_analysis.png': '오류 분석 (클래스별 정확도, 실패케이스)'
    }
    
    for i, filename in enumerate(plot_files, 1):
        desc = descriptions.get(filename, '')
        print(f"  {i:2d}. {filename:<30} - {desc}")
    
    print(f"\n모든 시각화가 'performance_plots/' 디렉토리에 저장되었습니다.")
    print(f"총 {len(plot_files)}개의 종합적인 SimCLR 모델 분석 차트가 생성되었습니다.")
    print("="*60)