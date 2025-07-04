import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # 진행 상황 바를 보여주는 라이브러리
import math
from torch.amp import autocast, GradScaler

# TensorFlow 메시지 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- AMP 관련 추가/변경 ---
from torch.cuda.amp import autocast, GradScaler # 새로 임포트

# 같은 src 폴더에 있는 model.py와 dataset.py에서 클래스 불러오기
from model import SimCLRVIT
from dataset import StanfordDogsDataset, get_simclr_transforms

# --- 하이퍼파라미터 설정 ---
# 이 값들은 나중에 실험을 통해 최적화할 수 있습니다.
BATCH_SIZE = 128  # GPU 메모리 최대 활용을 위해 크게 설정
IMAGE_SIZE = 224 # 이미지 크기 (ViT 모델 입력 크기에 맞춰야 함)
OUT_DIM = 128 # SimCLR 임베딩 벡터의 차원
EPOCHS = 20 # 총 훈련 에폭 수 (시간이 오래 걸리므로 초기엔 작게 시작)
LEARNING_RATE = 3e-4 # 학습률
WEIGHT_DECAY = 1e-6 # 가중치 감소 (L2 정규화 효과)
TEMPERATURE = 0.07 # SimCLR Loss 계산 시 사용되는 온도 파라미터 (논문 권장 값)

DATA_ROOT = 'training/Images' # 데이터셋 경로

SAVE_PATH = 'models/simclr_vit_dog_model.pth' # 모델 가중치 저장경로


# --- Contrastive Loss (NT-Xent Loss) 구현 ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2) # 코사인 유사도 계산 함수

    def forward(self, z_i, z_j):
        """ㄴ
        Args:
            z_i (torch.Tensor): 첫 번째 증강 뷰의 임베딩 텐서 [batch_size, out_dim]
            z_j (torch.Tensor): 두 번째 증강 뷰의 임베딩 텐서 [batch_size, out_dim]
        """
        batch_size = z_i.size(0)

        # 1. 두 임베딩 텐서를 결합하여 [2*batch_size, out_dim] 형태로 만듭니다.
        # 각 이미지 쌍 (anchor, positive)은 배치 내의 다른 모든 이미지와 negative 쌍을 이룹니다.
        z = torch.cat((z_i, z_j), dim=0) # [2*batch_size, out_dim]

        # 2. 모든 임베딩 쌍 간의 코사인 유사도를 계산합니다.
        # [2*batch_size, 1, out_dim]과 [1, 2*batch_size, out_dim]으로 확장하여 브로드캐스팅합니다.
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) # [2*batch_size, 2*batch_size]

        # 3. 'self-similarity' (같은 임베딩 간의 유사도)를 제거합니다.
        # 즉, 대각선 값을 0으로 만듭니다 (log(1) = 0 이므로).
        logits = sim / self.temperature # 온도 파라미터로 나누어 스케일 조정
        logits_mask = torch.eye(2 * batch_size).bool().to(self.device) # 대각선이 True인 마스크
        logits = logits.masked_fill_(logits_mask, -1e9) # 대각선 값에 매우 작은 값 할당 (로그 시 0이 되도록)

        # 4. Positive 샘플(같은 이미지의 두 증강 뷰)에 대한 마스크를 생성합니다.
        # (i, j) = (z_i의 임베딩, z_j의 임베딩)
        # 즉, (0, batch_size), (1, batch_size+1), ..., (batch_size-1, 2*batch_size-1)
        # (z_j의 임베딩, z_i의 임베딩)
        # 즉, (batch_size, 0), (batch_size+1, 1), ..., (2*batch_size-1, batch_size-1)
        labels = torch.arange(batch_size, dtype=torch.long).to(self.device)
        labels = torch.cat([labels, labels], dim=0) # [0,1,..,N-1, 0,1,..,N-1]
        
        # positive_mask는 자기 자신을 제외한 positive 쌍의 위치에 True
        positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool).to(self.device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        # negative_mask는 positive_mask와 logits_mask를 제외한 모든 위치에 True
        negative_mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool).to(self.device)
        negative_mask = negative_mask.masked_fill_(positive_mask | logits_mask, False)

        # 5. Contrastive Loss 계산
        # NCE Loss (Normalized Temperature-scaled Cross Entropy Loss)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Positive 쌍의 로그 확률만 추출하여 평균을 냅니다.
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-9) # +1e-9는 0으로 나누는 것 방지

        loss = -mean_log_prob_pos.mean() # 최종 Loss는 음수 로그 확률의 평균

        return loss

# --- 훈련 함수 정의 ---
def train_simclr():
    # 1. 장치 설정 (GPU 또는 CPU)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA(GPU)가 감지되지 않았습니다. GPU 환경에서 실행해주세요!")
    device = torch.device("cuda")
    print(f"훈련에 사용할 장치: {device} ({torch.cuda.get_device_name(0)})")

    # GPU 연산 확인용 코드
    print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
    print("[DEBUG] torch.cuda.current_device():", torch.cuda.current_device())
    print("[DEBUG] torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

    # 2. 데이터셋 및 데이터로더 준비
    print(f"데이터셋 경로: {DATA_ROOT}")
    simclr_transforms = get_simclr_transforms(IMAGE_SIZE)
    # sample_ratio를 0.50 (50%)로 설정합니다.
    dataset = StanfordDogsDataset(root_dir=DATA_ROOT, transform=simclr_transforms, sample_ratio=0.50)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)  # GPU 사용시 pin_memory=True
    # num_workers는 CPU 코어 수의 절반 정도로 설정하는 것이 일반적입니다.

    # 3. 모델 인스턴스 생성 및 GPU로 이동
    model = SimCLRVIT(out_dim=OUT_DIM)
    model.to(device)

    # 4. Loss 함수 및 옵티마이저 설정
    criterion = NTXentLoss(temperature=TEMPERATURE, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # (선택 사항) 학습률 스케줄러: cosine annealing scheduler
    # 총 학습 스텝 수 계산
    total_steps = len(dataloader) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- AMP 관련 추가/변경 ---
    scaler = GradScaler() # GradScaler 인스턴스 생성

    # 5. 훈련 루프 시작
    print(f"\nSimCLR 훈련 시작! (에폭 수: {EPOCHS}, 배치 크기: {BATCH_SIZE})")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train() # 모델을 훈련 모드로 설정
        total_loss = 0
        
        # tqdm으로 진행 상황 바 표시
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (img1, img2, labels) in enumerate(loop):
            # 데이터를 GPU로 이동
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)

            # [DEBUG] 실제 텐서와 모델이 GPU에 있는지 확인
            if batch_idx == 0:
                print("[DEBUG] img1.device:", img1.device)
                print("[DEBUG] model.device:", next(model.parameters()).device)

            # 옵티마이저의 기울기 초기화
            optimizer.zero_grad()

            # --- AMP 관련 추가/변경 ---
            # autocast 컨텍스트 내에서 forward pass 실행 (자동으로 float16으로 변환)
            with autocast():
                z_i = model(img1)
                z_j = model(img2)
                loss = criterion(z_i, z_j)

            # --- AMP 관련 추가/변경 ---
            # 스케일러를 사용하여 Loss 스케일링 후 역전파
            scaler.scale(loss).backward()#8#*
            scaler.step(optimizer)
            scaler.update()
            
            # 학습률 스케줄러 업데이트 (에폭당 또는 스텝당)
            scheduler.step()

            total_loss += loss.item() # Loss 누적

            # --- GPU 메모리 사용량 출력 ---
            # if batch_idx % 10 == 0:
            #     allocated = torch.cuda.memory_allocated(device) / 1024**2
            #     reserved = torch.cuda.memory_reserved(device) / 1024**2
            #     print(f"[DEBUG][Batch {batch_idx}] GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

            # tqdm 진행 바에 현재 Loss 표시
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

        # 모델 저장 (가장 낮은 Loss를 달성했을 때)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"-> 모델 저장됨: {SAVE_PATH} (Loss: {avg_loss:.4f})")

    print("\nSimCLR 훈련 완료!")

def find_max_batch_size(start=256, step=128, max_batch=512):
    """
    GPU OOM이 날 때까지 배치 크기를 자동으로 늘려가며 최대 배치 크기를 찾는 함수
    """
    import time
    print("\n[Auto BatchSize Search] 시작...")
    batch_size = start
    last_success = start
    while batch_size <= max_batch:
        try:
            print(f"  시도 중: batch_size={batch_size}")
            simclr_transforms = get_simclr_transforms(IMAGE_SIZE)
            dataset = StanfordDogsDataset(root_dir=DATA_ROOT, transform=simclr_transforms, sample_ratio=0.50)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
            model = SimCLRVIT(out_dim=OUT_DIM).to("cuda")
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = NTXentLoss(temperature=TEMPERATURE, device="cuda")
            scaler = GradScaler()
            img1, img2, labels = next(iter(dataloader))
            img1 = img1.to("cuda")
            img2 = img2.to("cuda")
            torch.cuda.synchronize()
            start_time = time.time()
            with autocast():
                z_i = model(img1)
                z_j = model(img2)
                loss = criterion(z_i, z_j)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            allocated = torch.cuda.memory_allocated("cuda") / 1024**2
            reserved = torch.cuda.memory_reserved("cuda") / 1024**2
            print(f"  성공: batch_size={batch_size} | GPU Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, 연산시간: {elapsed:.2f}s")
            last_success = batch_size
            batch_size += step
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  OOM 발생! 최대 안전 배치 크기: {last_success}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    print("[Auto BatchSize Search] 완료\n")
    return last_success

# --- 스크립트 실행 ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    # 자동 배치 사이즈 탐색 실행 (start, step, max_batch 모두 명확히 지정)
    # max_batch = find_max_batch_size(start=256, step=128, max_batch=512)
    # print(f"[INFO] 추천 배치 크기: {max_batch}")
    # BATCH_SIZE를 자동으로 조정하려면 아래 주석 해제
    # BATCH_SIZE = max_batch
    train_simclr()