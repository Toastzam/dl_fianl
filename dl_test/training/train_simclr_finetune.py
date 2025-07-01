import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 하이퍼파라미터 및 경로 설정 ---
DATA_DIR = 'training/Images'  # DB 이미지 폴더 (클래스별 하위폴더 가능, 라벨 미사용)
PRETRAINED_MODEL_PATH = 'models/simclr_vit_dog_model.pth'  # 기존 SimCLR 사전학습 모델
FINETUNED_MODEL_PATH = 'models/simclr_vit_dog_model_finetuned.pth'  # 파인튜닝 후 저장 경로
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 데이터셋 및 데이터로더 ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 라벨 없이 이미지 전체를 사용 (ImageFolder는 폴더명을 라벨로 사용하지만, SimCLR는 라벨 미사용)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- SimCLR 모델 정의 (예시: ResNet 기반) ---
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super().__init__()
        self.encoder = base_model
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        if isinstance(h, tuple):  # ViT 등 일부 모델은 (features, ) 반환
            h = h[0]
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return z

# --- 사전학습 모델 불러오기 ---
base_model = models.resnet18(pretrained=False)
base_model.fc = nn.Identity()
model = SimCLR(base_model, out_dim=128)
if os.path.exists(PRETRAINED_MODEL_PATH):
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    print(f"사전학습 SimCLR 모델 로드: {PRETRAINED_MODEL_PATH}")
else:
    print("사전학습 모델이 없습니다. 랜덤 초기화로 진행합니다.")
model = model.to(DEVICE)

# --- SimCLR NT-Xent Loss (간단 구현) ---
def nt_xent_loss(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T)
    labels = torch.arange(z.size(0)).to(z.device)
    mask = torch.eye(z.size(0), dtype=torch.bool).to(z.device)
    similarity = similarity / temperature
    similarity = similarity.masked_fill(mask, -9e15)
    loss = nn.CrossEntropyLoss()(similarity, labels)
    return loss

# --- Optimizer ---
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 파인튜닝 루프 ---
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # SimCLR: 같은 이미지를 두 번 augment해서 쌍 생성
        images1 = images
        images2 = images[torch.randperm(images.size(0))]
        images1, images2 = images1.to(DEVICE), images2.to(DEVICE)
        z1 = model(images1)
        z2 = model(images2)
        z = torch.cat([z1, z2], dim=0)
        loss = nt_xent_loss(z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# --- 파인튜닝된 모델 저장 ---
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"파인튜닝된 SimCLR 모델 저장 완료: {FINETUNED_MODEL_PATH}")