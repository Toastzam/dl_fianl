# =====================
# SimCLR Linear Evaluation & Fine-tuning Protocols
# =====================
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from model import SimCLRVIT
from dataset import StanfordDogsDataset, get_simclr_transforms
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision을 위한 import
import torchvision.transforms as T  # GPU 데이터 증강을 위한 추가 import
import os

# PyTorch 성능 최적화 설정
torch.backends.cudnn.benchmark = True  # cuDNN 자동 최적화
torch.backends.cudnn.deterministic = False  # 성능 우선 (재현성보다)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # GPU 커널 비동기 실행

print('1. 데이터셋 준비 시작')
data_dir = r'C:\dl_final\dl_test\training\Images'
num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
full_dataset = StanfordDogsDataset(root_dir=data_dir, transform=get_simclr_transforms(224), sample_ratio=1.0)
print('1. 데이터셋 준비 완료')


print('2. 데이터 분할 시작')
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print('2. 데이터 분할 완료')
print('2-1. DataLoader 생성 시작')
# CPU 병목현상 해결을 위한 최적화 (Windows 호환성 고려)
train_loader = DataLoader(
    train_dataset, 
    batch_size=512, 
    shuffle=True, 
    num_workers=0,  # Windows 호환성
    pin_memory=True,
    drop_last=True  # 마지막 불완전한 배치 제거로 일관성 향상
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=512, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True,
    drop_last=False
)
print('2-1. DataLoader 생성 완료')



print('3. 디바이스 설정 및 Encoder 불러오기 시작')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = SimCLRVIT(out_dim=128)
encoder.load_state_dict(torch.load(r'C:\dl_final\models\simclr_vit_dog_model_finetuned_v1.pth', map_location=device))
encoder.to(device)
encoder.train()  # GPU 메모리 사용량을 늘리기 위해 train 모드로 변경
print('3. 디바이스 설정 및 Encoder 불러오기 완료')


print('4. Linear Classifier 정의')
classifier = nn.Linear(128, num_classes).to(device)
print('4. Linear Classifier 완료')


print('5. Linear Evaluation 준비')
# GPU 메모리 사용량을 늘리기 위해 Encoder도 함께 학습 (Fine-tuning)
for p in encoder.parameters():
    p.requires_grad = True  # Encoder도 학습 가능하게 설정
    
# Encoder와 Classifier 모두 최적화
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()), 
    lr=1e-4  # Fine-tuning에는 더 작은 learning rate 사용
)
criterion = nn.CrossEntropyLoss()

# GPU 메모리 사용량 모니터링을 위한 설정
num_epochs = 10
best_acc = 0
# Mixed Precision을 위한 GradScaler 초기화
scaler = GradScaler()
print(f'5. Linear Evaluation 준비 완료 - GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f}GB')

# --- CSV 기록용 ---
import csv


print('6. Linear Evaluation 학습 루프 진입')

# ================= 학습 루프 복구 + 안전 print 추가 =================
print('6. Linear Evaluation 학습 루프 진입')
linear_logfile = 'linear_eval_log.csv'
with open(linear_logfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for epoch in range(num_epochs):
        print(f'6-1. Epoch {epoch+1} 시작')
        classifier.train()
        total, correct, total_loss = 0, 0, 0
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            try:
                # 비동기 GPU 전송으로 CPU-GPU 병목 감소
                img1 = img1.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Mixed Precision으로 GPU 메모리 효율성 향상
                with autocast():
                    feats = encoder(img1)
                    logits = classifier(feats)
                    loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total += labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
                if batch_idx % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated()/1024**3
                    # GPU 사용률도 함께 확인
                    torch.cuda.synchronize()  # GPU 연산 완료 대기
                    print(f"[Train][Epoch {epoch+1}][Batch {batch_idx}] Loss: {loss.item():.4f}, GPU Memory: {gpu_memory:.2f}GB")
            except Exception as e:
                print(f"[Train][Epoch {epoch+1}][Batch {batch_idx}] 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        train_acc = correct / total
        train_loss = total_loss / total
        print(f'6-2. Epoch {epoch+1} train 끝')
        # Validation
        classifier.eval()
        total, correct, total_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(val_loader):
                try:
                    # 비동기 GPU 전송으로 CPU-GPU 병목 감소
                    img1 = img1.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    feats = encoder(img1)
                    logits = classifier(feats)
                    loss = criterion(logits, labels)
                    total += labels.size(0)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total_loss += loss.item() * labels.size(0)
                    if batch_idx % 10 == 0:
                        print(f"[Val][Epoch {epoch+1}][Batch {batch_idx}] Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"[Val][Epoch {epoch+1}][Batch {batch_idx}] 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        val_acc = correct / total
        val_loss = total_loss / total
        print(f"[Linear Eval][Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), 'linear_classifier_best.pth')
        print(f'6-3. Epoch {epoch+1} 종료')

# 7. Semi-supervised 실험: train_dataset 일부만 사용 (5%)
semi_size = int(len(train_dataset) * 0.05)
semi_train, _ = random_split(train_dataset, [semi_size, len(train_dataset)-semi_size])
# Windows multiprocessing 문제를 피하기 위해 num_workers=0 사용
# GPU 메모리를 더 많이 사용하기 위해 배치 사이즈 증가
semi_loader = DataLoader(semi_train, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)



# --- Semi-supervised Linear Evaluation ---
print(f"\n[SEMI-SUPERVISED] Training with only {semi_size} samples ({100*semi_size/len(train_dataset):.2f}%) of labeled data.")
classifier_semi = nn.Linear(128, num_classes).to(device)
for p in encoder.parameters():
    p.requires_grad = False
optimizer_semi = torch.optim.Adam(classifier_semi.parameters(), lr=1e-3)
criterion_semi = nn.CrossEntropyLoss()
num_epochs_semi = 20
best_acc_semi = 0
semi_logfile = 'semi_eval_log.csv'
with open(semi_logfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for epoch in range(num_epochs_semi):
        classifier_semi.train()
        total, correct, total_loss = 0, 0, 0
        for batch_idx, (img1, img2, labels) in enumerate(semi_loader):
            try:
                img1, labels = img1.to(device), labels.to(device)
                with torch.no_grad():
                    feats = encoder(img1)
                logits = classifier_semi(feats)
                loss = criterion_semi(logits, labels)
                optimizer_semi.zero_grad()
                loss.backward()
                optimizer_semi.step()
                total += labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
                if batch_idx % 10 == 0:
                    print(f"[Semi][Epoch {epoch+1}][Batch {batch_idx}] Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"[Semi][Epoch {epoch+1}][Batch {batch_idx}] 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        train_acc = correct / total
        train_loss = total_loss / total
        # Validation
        classifier_semi.eval()
        total, correct, total_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(val_loader):
                try:
                    img1, labels = img1.to(device), labels.to(device)
                    feats = encoder(img1)
                    logits = classifier_semi(feats)
                    loss = criterion_semi(logits, labels)
                    total += labels.size(0)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total_loss += loss.item() * labels.size(0)
                    if batch_idx % 10 == 0:
                        print(f"[Semi Val][Epoch {epoch+1}][Batch {batch_idx}] Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"[Semi Val][Epoch {epoch+1}][Batch {batch_idx}] 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        val_acc = correct / total
        val_loss = total_loss / total
        print(f"[Semi-supervised][Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        if val_acc > best_acc_semi:
            best_acc_semi = val_acc
            torch.save(classifier_semi.state_dict(), 'linear_classifier_semi_best.pth')

print('실험 완료!')

# ================= GPU 메모리 최적화 실험 (주석처리) =================
# print('\n=== GPU 메모리 최적화 실험 시작 ===')

# # 더 큰 배치 사이즈로 테스트 (GPU 메모리 허용하는 한도까지)
# try:
#     print('더 큰 배치 사이즈로 DataLoader 재생성 중...')
#     # 배치 사이즈를 512로 증가하여 GPU 메모리 최대 활용 시도
#     train_loader_large = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
#     val_loader_large = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
#     
#     # 테스트 배치로 GPU 메모리 사용량 확인
#     test_batch = next(iter(train_loader_large))
#     img1, img2, labels = test_batch
#     img1, labels = img1.to(device), labels.to(device)
#     
#     with torch.no_grad():
#         feats = encoder(img1)
#     
#     gpu_memory_after = torch.cuda.memory_allocated()/1024**3
#     print(f'배치 사이즈 512 테스트 성공! GPU 메모리 사용량: {gpu_memory_after:.2f}GB')
#     
#     # 성공하면 더 큰 DataLoader 사용
#     train_loader = train_loader_large
#     val_loader = val_loader_large
#     print('배치 사이즈 512로 업데이트 완료')
#     
# except RuntimeError as e:
#     if "out of memory" in str(e):
#         print(f'배치 사이즈 512는 GPU 메모리 부족: {e}')
#         print('기존 배치 사이즈 256 유지')
#     else:
#         print(f'예상치 못한 오류: {e}')

# print('=== GPU 메모리 최적화 실험 완료 ===\n')
