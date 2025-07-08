니# =====================
# SimCLR Embedding Performance Visualization (Advanced)
# =====================
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix, top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from model import SimCLRVIT
from dataset import StanfordDogsDataset, get_simclr_transforms
from torch.utils.data import DataLoader
import seaborn as sns

# 1. 모델 로드
model = SimCLRVIT(out_dim=128)
model.load_state_dict(torch.load('models/simclr_vit_dog_model_finetuned_v1.pth'))
model.eval()
model.cuda()

# 2. 데이터셋 준비 (라벨 포함)
val_dataset = StanfordDogsDataset(root_dir=r'C:\dl_final\dl_fianl\dl_test\training\images', transform=get_simclr_transforms(224), sample_ratio=1.0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

all_embeds = []
all_labels = []

with torch.no_grad():
    for img1, img2, labels in val_loader:
        img1 = img1.cuda()
        embeds = model(img1).cpu().numpy()
        all_embeds.append(embeds)
        all_labels.append(labels.numpy())

all_embeds = np.concatenate(all_embeds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
num_classes = len(np.unique(all_labels))



# 3. t-SNE 시각화 (클래스별 산점도)
tsne = TSNE(n_components=2, random_state=42)
embeds_2d = tsne.fit_transform(all_embeds)
plt.figure(figsize=(10,8))
scatter = plt.scatter(embeds_2d[:,0], embeds_2d[:,1], c=all_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, label='Class Label')
plt.title('t-SNE of SimCLR Embeddings')
plt.show()

# 3-1. 클러스터링 결과 시각화 (KMeans)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=num_classes, random_state=42)
cluster_labels = kmeans.fit_predict(all_embeds)
plt.figure(figsize=(10,8))
plt.scatter(embeds_2d[:,0], embeds_2d[:,1], c=cluster_labels, cmap='tab20', alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.title('t-SNE with KMeans Clustering')
plt.show()

# 3-2. 데이터 증강 효과 시각화 (원본 vs 증강)
sample_idx = np.random.choice(len(val_dataset), size=5, replace=False)
fig, axes = plt.subplots(5, 2, figsize=(6, 14))
for i, idx in enumerate(sample_idx):
    orig_img, _, _ = val_dataset[idx]
    aug_img, _, _ = val_dataset.__getitem__(idx)
    axes[i,0].imshow(np.transpose(orig_img.cpu().numpy(), (1,2,0)))
    axes[i,0].set_title('Original')
    axes[i,0].axis('off')
    axes[i,1].imshow(np.transpose(aug_img.cpu().numpy(), (1,2,0)))
    axes[i,1].set_title('Augmented')
    axes[i,1].axis('off')
plt.suptitle('Data Augmentation Effect (Random Samples)')
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()

# 4. 클래스별 centroid 시각화
unique_labels = np.unique(all_labels)
centroids = np.array([all_embeds[all_labels==lbl].mean(axis=0) for lbl in unique_labels])
centroids_2d = tsne.fit_transform(centroids)
plt.figure(figsize=(8,6))
plt.scatter(centroids_2d[:,0], centroids_2d[:,1], c=unique_labels, cmap='tab20', s=120, edgecolor='k')
for i, lbl in enumerate(unique_labels):
    plt.text(centroids_2d[i,0], centroids_2d[i,1], str(lbl), fontsize=9)
plt.title('t-SNE of Class Centroids')
plt.show()

# 5. 임베딩 norm boxplot (클래스별)
norms = np.linalg.norm(all_embeds, axis=1)
plt.figure(figsize=(12,6))
data = [norms[all_labels==lbl] for lbl in unique_labels]
plt.boxplot(data, labels=unique_labels)
plt.xlabel('Class Label')
plt.ylabel('Embedding Norm')
plt.title('Embedding Norm by Class')
plt.show()

# 6. 최근접 이웃 기반 confusion matrix, Top-1/5 accuracy
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(all_embeds)
np.fill_diagonal(sim_matrix, -np.inf)  # 자기 자신 제외
top1_idx = np.argmax(sim_matrix, axis=1)
top1_pred = all_labels[top1_idx]
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(all_labels, top1_pred, labels=unique_labels)
plt.figure(figsize=(10,8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Nearest Neighbor)')
plt.show()

# Top-1, Top-5 accuracy
top5_idx = np.argpartition(sim_matrix, -5, axis=1)[:,-5:]
top5_pred = all_labels[top5_idx]
top1_acc = accuracy_score(all_labels, top1_pred)
top5_acc = np.mean([label in top5 for label, top5 in zip(all_labels, top5_pred)])
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")


# 7. Positive/Negative 쌍 유사도 분포
pos_sims = sim_matrix[all_labels[:,None]==all_labels[None,:]]
neg_sims = sim_matrix[all_labels[:,None]!=all_labels[None,:]]
plt.figure(figsize=(8,5))
plt.hist(pos_sims, bins=50, alpha=0.6, label='Positive')
plt.hist(neg_sims, bins=50, alpha=0.6, label='Negative')
plt.legend()
plt.title('Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Count')
plt.show()

# 7-1. 코사인 유사도 행렬 히트맵
plt.figure(figsize=(10,8))
sns.heatmap(sim_matrix, cmap='coolwarm', center=0, cbar_kws={'label': 'Cosine Similarity'})
plt.title('Cosine Similarity Matrix Heatmap')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')
plt.tight_layout()
plt.show()

# 8. ROC Curve (이진 분류/쌍별 평가)
# 8. ROC Curve (이진 분류/쌍별 평가)
y_true = (all_labels[:,None]==all_labels[None,:]).astype(int).flatten()
y_score = sim_matrix.flatten()
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Pairwise Similarity)')
plt.legend()
plt.show()


# 9. 학습 곡선 시각화 (loss/acc, csv 자동 연동)

import pandas as pd
import os
def plot_csv_log(csv_path, title_prefix):
    if not os.path.exists(csv_path):
        print(f'{csv_path} not found.')
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Linear Eval log
plot_csv_log('linear_eval_log.csv', 'Linear Eval')
# Semi-supervised log
plot_csv_log('semi_eval_log.csv', 'Semi-supervised')

# 7. Confusion Matrix (임베딩 최근접 이웃 기반)
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=2, metric='cosine').fit(all_embeds)
distances, indices = knn.kneighbors(all_embeds)
# indices[:,0] == self, indices[:,1] == nearest neighbor
pred_labels = all_labels[indices[:,1]]
cm = confusion_matrix(all_labels, pred_labels, labels=range(num_classes))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label (NN)')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Nearest Neighbor)')
plt.show()

# 8. Top-1, Top-5 Accuracy (임베딩 최근접 이웃 기반)
from sklearn.metrics import accuracy_score
top1_acc = accuracy_score(all_labels, pred_labels)
# Top-5: for each sample, check if true label in 5 nearest neighbors (excluding self)
knn5 = NearestNeighbors(n_neighbors=6, metric='cosine').fit(all_embeds)
_, indices5 = knn5.kneighbors(all_embeds)
top5_hits = [all_labels[i] in all_labels[indices5[i,1:]] for i in range(len(all_labels))]
top5_acc = np.mean(top5_hits)
print(f"Top-1 Accuracy (NN): {top1_acc:.4f}")
print(f"Top-5 Accuracy (NN): {top5_acc:.4f}")

# 9. ROC Curve (이진 분류/쌍별 평가)
y_true = (all_labels[:,None]==all_labels[None,:]).astype(int).flatten()
y_score = sim_matrix.flatten()
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Pairwise Similarity)')
plt.legend()
plt.show()

# 4. Positive/Negative 쌍 유사도 분포 (예시)
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(all_embeds)
# (라벨이 같으면 positive, 다르면 negative)
pos_sims = sim_matrix[all_labels[:,None]==all_labels[None,:]]
neg_sims = sim_matrix[all_labels[:,None]!=all_labels[None,:]]
plt.hist(pos_sims, bins=50, alpha=0.6, label='Positive')
plt.hist(neg_sims, bins=50, alpha=0.6, label='Negative')
plt.legend()
plt.title('Cosine Similarity Distribution')
plt.show()

# 5. ROC Curve (이진 분류/쌍별 평가)
y_true = (all_labels[:,None]==all_labels[None,:]).astype(int).flatten()
y_score = sim_matrix.flatten()
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Pairwise Similarity)')
plt.legend()
plt.show()