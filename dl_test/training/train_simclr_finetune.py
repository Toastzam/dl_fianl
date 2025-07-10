import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision 최적화
from PIL import Image
import glob

# SimCLR용 커스텀 Dataset 클래스 (단일 폴더에서 이미지 로드)
class SimCLRDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 지원하는 이미지 확장자
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        print(f"총 {len(self.image_paths)}개의 이미지를 찾았습니다.")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # SimCLR는 라벨을 사용하지 않으므로 더미 라벨 0
        except Exception as e:
            print(f"이미지 로드 오류: {image_path}, 에러: {e}")
            # 오류 발생 시 다음 이미지로 대체
            return self.__getitem__((idx + 1) % len(self.image_paths))

# Windows 멀티프로세싱 호환을 위한 collate_fn 함수 정의
def pil_collate_fn(batch):
    # PIL 이미지와 라벨을 분리
    images, labels = zip(*batch)
    return list(images), list(labels)  # PIL 이미지 리스트와 라벨 리스트 반환
from tqdm import tqdm
from model import SimCLRVIT

# TensorFlow 메시지 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 하이퍼파라미터 및 경로 설정 ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
DATA_DIR = r'C:\dl_final\training\Images\all'  # DB 이미지 폴더 (실제 이미지가 있는 all 폴더)
PRETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, 'simclr_vit_dog_model.pth')  # 기존 SimCLR 사전학습 모델
# FINETUNED_MODEL_PATH는 아래에서 동적으로 생성

# === 실험 설정 (Loss < 1.0 목표) ===
EXPERIMENT_CONFIG = {
    "batch_size": 64,        # 128 → 64로 감소 (더 안정적인 학습)
    "epochs": 80,            # 60 → 80로 증가 (더 많은 학습)
    "lr": 5e-5,              # 1e-4 → 5e-5로 감소 (더 안정적인 학습)
    "temperature": 0.1,      # 0.2 → 0.1로 감소 (더 strict한 대조 학습)
    "weight_decay": 1e-4,    # 1e-6 → 1e-4로 증가 (더 강한 정규화)
    "warmup_epochs": 5,      # 워밍업 에폭 추가
}

BATCH_SIZE = EXPERIMENT_CONFIG["batch_size"]
EPOCHS = EXPERIMENT_CONFIG["epochs"]
LR = EXPERIMENT_CONFIG["lr"]
TEMPERATURE = EXPERIMENT_CONFIG["temperature"]
WEIGHT_DECAY = EXPERIMENT_CONFIG["weight_decay"]
WARMUP_EPOCHS = EXPERIMENT_CONFIG["warmup_epochs"]
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"=== 실험 설정 (Loss < 1.0 목표) ===")
print(f"배치 크기: {BATCH_SIZE}, 에폭: {EPOCHS}, 학습률: {LR}")
print(f"온도: {TEMPERATURE}, 가중치 감쇠: {WEIGHT_DECAY}, 워밍업: {WARMUP_EPOCHS}")
print(f"===============================")

# --- 저장 폴더 ---
# all 폴더 생성 제거 (불필요)
# os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    # --- 버전 넘버로 저장 파일명 자동 생성 ---
    import re
    import glob
    # 기존 finetuned 모델 파일들 중 버전 넘버 추출 (절대경로 사용)
    finetuned_files = glob.glob(os.path.join(MODELS_DIR, 'simclr_vit_dog_model_finetuned_v*.pth'))
    print(f"[DEBUG] 찾은 finetuned 파일들: {finetuned_files}")
    
    version_numbers = []
    for f in finetuned_files:
        match = re.search(r"finetuned_v(\d+)\.pth", f)
        if match:
            version_numbers.append(int(match.group(1)))
            print(f"[DEBUG] 파일: {f}, 버전: {match.group(1)}")
    
    next_version = max(version_numbers) + 1 if version_numbers else 1
    print(f"[DEBUG] 다음 버전 번호: {next_version}")
    FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, f"simclr_vit_dog_model_finetuned_v{next_version}.pth")

    # --- 데이터셋 및 데이터로더 ---
    # 더 강력한 데이터 증강 (Loss < 1.0 목표)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),  # 0.2 → 0.08 (더 강한 crop)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomRotation(degrees=10),  # 회전 증강 추가
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 라벨 없이 이미지 전체를 사용 (커스텀 Dataset으로 단일 폴더에서 이미지 로드)
    # Dataset에는 transform을 적용하지 않고 PIL 이미지를 반환하도록 설정
    dataset = SimCLRDataset(DATA_DIR, transform=None)
    print(f"데이터셋 로드 완료: {len(dataset)}개 이미지")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows 멀티프로세싱 호환성을 위해 0으로 설정
        pin_memory=False,  # PIL 이미지는 pin_memory=False로 설정
        persistent_workers=False,  # num_workers=0일 때 False
        drop_last=True,
        collate_fn=pil_collate_fn  # PIL 이미지 처리를 위한 커스텀 collate function
    )

    # --- 데이터/정규화/샘플 체크 ---
    # import matplotlib.pyplot as plt
    import numpy as np
    # def show_img(tensor_img, title=""):
    #     img = tensor_img.clone().detach().cpu().numpy()
    #     img = np.transpose(img, (1,2,0))
    #     img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    #     img = np.clip(img, 0, 1)
    #     plt.imshow(img)
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.show()

    # 데이터 샘플 체크
    sample_img, _ = dataset[0]  # PIL 이미지 반환
    sample_tensor = transform(sample_img)  # PIL 이미지를 tensor로 변환 및 정규화
    print(f"[데이터 샘플] 이미지 크기: {sample_img.size}, 텐서 shape: {sample_tensor.shape}")
    print(f"[데이터 샘플] min: {sample_tensor.min().item():.3f}, max: {sample_tensor.max().item():.3f}, mean: {sample_tensor.mean().item():.3f}, std: {sample_tensor.std().item():.3f}")
    # show_img(sample_tensor, title="정규화 후 샘플 이미지")

    # --- GPU 메모리 및 환경 확인 ---
    if not torch.cuda.is_available():
        print("경고: CUDA(GPU)가 감지되지 않았습니다. CPU로 실행됩니다.")
    else:
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- SimCLR ViT 모델 불러오기 (model.py 기반) ---
    model = SimCLRVIT(out_dim=128)
    # 가장 최근 finetuned 모델이 있으면 불러오기 (절대경로 사용, 버전 없는 파일도 포함)
    finetuned_files = sorted(
        glob.glob(os.path.join(MODELS_DIR, 'simclr_vit_dog_model_finetuned*.pth')),
        reverse=True
    )
    if finetuned_files:
        model.load_state_dict(torch.load(finetuned_files[0], map_location=DEVICE))
        print(f"이전 파인튜닝 모델 로드: {finetuned_files[0]}")
    elif os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        print(f"사전학습 SimCLR 모델 로드: {PRETRAINED_MODEL_PATH}")
    else:
        print("사전학습 모델이 없습니다. 랜덤 초기화로 진행합니다.")
    model = model.to(DEVICE)

    # --- SimCLR NT-Xent Loss (온도 파라미터 최적화) ---
    def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
        batch_size = z1.size(0)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        similarity_matrix = torch.matmul(z, z.T)  # (2N, 2N)
        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        similarity_matrix = similarity_matrix / temperature
        # Mixed Precision 호환을 위해 마스크 값 조정
        mask_value = -1e4 if z.dtype == torch.float16 else -9e15
        similarity_matrix = similarity_matrix.masked_fill(mask, mask_value)

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

    # --- Optimizer & Scheduler (개선된 설정) ---
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # 워밍업 + Cosine Annealing Scheduler
    total_steps = len(dataloader) * EPOCHS
    warmup_steps = len(dataloader) * WARMUP_EPOCHS
    
    def lr_lambda(step):
        if step < warmup_steps:
            # 워밍업: 선형 증가
            return step / warmup_steps
        else:
            # Cosine Annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"총 스텝: {total_steps}, 워밍업 스텝: {warmup_steps}")
    
    # Mixed Precision을 위한 GradScaler
    scaler = GradScaler()

    # --- 파인튜닝 루프 (개선된 버전) ---
    model.train()
    best_loss = float('inf')
    current_save_path = FINETUNED_MODEL_PATH  # 초기 저장 경로
    patience = 10  # Early stopping patience
    patience_counter = 0
    loss_history = []
    
    print(f"\nSimCLR 파인튜닝 시작! (에폭 수: {EPOCHS}, 배치 크기: {BATCH_SIZE})")
    print(f"모델 저장 경로: {current_save_path}")
    print(f"목표: Loss < 1.0, Early Stopping Patience: {patience}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        
        for batch_idx, (images, _) in enumerate(loop):
            # SimCLR: 같은 이미지를 두 번 독립적으로 augment해서 쌍 생성 (PIL 이미지에 transform 적용)
            images1 = torch.stack([transform(img) for img in images])
            images2 = torch.stack([transform(img) for img in images])
            images1, images2 = images1.to(DEVICE, non_blocking=True), images2.to(DEVICE, non_blocking=True)

            # GPU 메모리 사용량 체크 (첫 번째 배치에서만)
            if batch_idx == 0 and epoch == 0:
                print(f"[DEBUG] images1.device: {images1.device}")
                print(f"[DEBUG] model.device: {next(model.parameters()).device}")

            # 옵티마이저 기울기 초기화
            optimizer.zero_grad()

            # Forward pass (Mixed Precision 비활성화하여 오버플로우 방지)
            z1 = model(images1)
            z2 = model(images2)
            loss = nt_xent_loss(z1, z2)

            # 로스값 nan/inf 체크
            if not torch.isfinite(loss):
                print("[경고] Loss가 비정상(nan/inf)입니다! 학습 중단.")
                break

            # 일반 역전파
            loss.backward()
            
            # Gradient Clipping 추가 (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 학습률 스케줄러 업데이트
            scheduler.step()

            total_loss += loss.item()
            
            # tqdm 진행 바에 현재 Loss와 학습률 표시
            current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # 개선된 로깅
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
        if avg_loss < 1.0:
            print(f"🎉 목표 달성! Loss < 1.0 (현재: {avg_loss:.4f})")
        
        # Best model 저장 (개선된 경우에만)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), current_save_path)
            print(f"-> 개선된 모델 저장: {current_save_path} (Loss: {avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping! {patience} 에폭 동안 개선 없음.")
                break

    print(f"\nSimCLR 파인튜닝 완료! 최종 Loss: {best_loss:.4f}")
    print(f"파인튜닝된 SimCLR 모델 저장 완료: {current_save_path}")
    print(f"Loss 히스토리: {loss_history[-5:]}")  # 마지막 5개 에폭만 표시