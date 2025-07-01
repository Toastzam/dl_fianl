import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Windows 멀티프로세싱 호환을 위한 collate_fn 함수 정의
def pil_collate_fn(batch):
    return tuple(zip(*batch))
from tqdm import tqdm
from model import SimCLRVIT

# TensorFlow 메시지 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 하이퍼파라미터 및 경로 설정 ---
DATA_DIR = 'training/Images'  # DB 이미지 폴더 (클래스별 하위폴더 가능, 라벨 미사용)
PRETRAINED_MODEL_PATH = 'models/simclr_vit_dog_model.pth'  # 기존 SimCLR 사전학습 모델
FINETUNED_MODEL_PATH = 'models/simclr_vit_dog_model_finetuned.pth'  # 파인튜닝 후 저장 경로
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 저장 폴더 ---
SAVE_DIR = 'training/Images/all'
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    # --- 데이터셋 및 데이터로더 ---
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 라벨 없이 이미지 전체를 사용 (ImageFolder는 폴더명을 라벨로 사용하지만, SimCLR는 라벨 미사용)
    dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=pil_collate_fn  # lambda 대신 함수명 사용 (Windows 호환)
    )

    # --- 데이터/정규화/샘플 체크 ---
    import matplotlib.pyplot as plt
    import numpy as np
    def show_img(tensor_img, title=""):
        img = tensor_img.clone().detach().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    # 데이터 샘플 체크
    sample_img, _ = dataset[0]
    sample_tensor = transform(sample_img)  # PIL 이미지를 tensor로 변환 및 정규화
    # print("[데이터 샘플] min:", sample_tensor.min().item(), "max:", sample_tensor.max().item(), "mean:", sample_tensor.mean().item(), "std:", sample_tensor.std().item())
    # show_img(sample_tensor, title="정규화 후 샘플 이미지")

    # --- SimCLR ViT 모델 불러오기 (model.py 기반) ---
    model = SimCLRVIT(out_dim=128)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        print(f"사전학습 SimCLR 모델 로드: {PRETRAINED_MODEL_PATH}")
    else:
        print("사전학습 모델이 없습니다. 랜덤 초기화로 진행합니다.")
    model = model.to(DEVICE)

    # --- SimCLR NT-Xent Loss (정상 구현) ---
    def nt_xent_loss(z1, z2, temperature=0.2):
        batch_size = z1.size(0)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        similarity_matrix = torch.matmul(z, z.T)  # (2N, 2N)
        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        similarity_matrix = similarity_matrix / temperature
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Positive pairs: i <-> i+N
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])

        # For each sample i, positive is at index i+N (for i in 0~N-1), and i-N (for i in N~2N-1)
        loss = 0
        for i in range(2 * batch_size):
            pos_idx = labels[i]
            logits = similarity_matrix[i]
            loss += nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([pos_idx]).to(z.device))
        loss = loss / (2 * batch_size)
        return loss

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- 파인튜닝 루프 ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # SimCLR: 같은 이미지를 두 번 독립적으로 augment해서 쌍 생성 (PIL 이미지에 transform 적용)
            images1 = torch.stack([transform(img) for img in images])
            images2 = torch.stack([transform(img) for img in images])
            images1, images2 = images1.to(DEVICE), images2.to(DEVICE)

            # 데이터 텐서 통계 체크
            # print(f"[Batch] min: {images1.min().item():.4f}, max: {images1.max().item():.4f}, mean: {images1.mean().item():.4f}, std: {images1.std().item():.4f}")

            z1 = model(images1)
            z2 = model(images2)

            # 모델 출력 통계 체크
            # print(f"[z1] min: {z1.min().item():.4f}, max: {z1.max().item():.4f}, mean: {z1.mean().item():.4f}, std: {z1.std().item():.4f}")
            # print(f"[z2] min: {z2.min().item():.4f}, max: {z2.max().item():.4f}, mean: {z2.mean().item():.4f}, std: {z2.std().item():.4f}")

            loss = nt_xent_loss(z1, z2)

            # 로스값 nan/inf 체크
            if not torch.isfinite(loss):
                print("[경고] Loss가 비정상(nan/inf)입니다! 학습 중단.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[Loss] {loss.item():.4f}")
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # --- 파인튜닝된 모델 저장 ---
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"파인튜닝된 SimCLR 모델 저장 완료: {FINETUNED_MODEL_PATH}")