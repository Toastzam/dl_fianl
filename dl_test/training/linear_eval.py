# =====================
# SimCLR Linear Evaluation & Fine-tuning Protocols
# =====================
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from model import SimCLRVIT
from dataset import StanfordDogsDataset, get_simclr_transforms
import os



print('1. 데이터셋 준비 시작')
data_dir = r'C:\dl_final\dl_fianl\dl_test\training\images'
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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
print('2-1. DataLoader 생성 완료')



print('3. 디바이스 설정 및 Encoder 불러오기 시작')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = SimCLRVIT(out_dim=128)
encoder.load_state_dict(torch.load('models/simclr_vit_dog_model_finetuned_v1.pth', map_location=device))
encoder.to(device)
encoder.eval()
print('3. 디바이스 설정 및 Encoder 불러오기 완료')


print('4. Linear Classifier 정의')
classifier = nn.Linear(128, num_classes).to(device)
print('4. Linear Classifier 완료')


print('5. Linear Evaluation 준비')
for p in encoder.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 20
best_acc = 0
print('5. Linear Evaluation 준비 완료')

# --- CSV 기록용 ---
import csv


print('6. Linear Evaluation 학습 루프 진입')
print('6-0. DataLoader에서 한 batch만 테스트')
for i, (img1, img2, label) in enumerate(train_loader):
    print(f"[DEBUG][train_loader] i={i}, img1 shape={img1.shape}, label={label}")
    if i > 10:
        break
print('6-0. DataLoader 테스트 완료')

# 기존 학습 루프는 주석 처리 (문제 진단 후 복구)
# linear_logfile = 'linear_eval_log.csv'
# with open(linear_logfile, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
#     for epoch in range(num_epochs):
#         print(f'6-1. Epoch {epoch+1} 시작')
#         classifier.train()
#         total, correct, total_loss = 0, 0, 0
#         for img1, img2, labels in train_loader:
#             img1, labels = img1.to(device), labels.to(device)
#             with torch.no_grad():
#                 feats = encoder(img1)
#             logits = classifier(feats)
#             loss = criterion(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total += labels.size(0)
#             correct += (logits.argmax(1) == labels).sum().item()
#             total_loss += loss.item() * labels.size(0)
#         train_acc = correct / total
#         train_loss = total_loss / total
#         print(f'6-2. Epoch {epoch+1} train 끝')
#         # Validation
#         classifier.eval()
#         total, correct, total_loss = 0, 0, 0
#         with torch.no_grad():
#             for img1, img2, labels in val_loader:
#                 img1, labels = img1.to(device), labels.to(device)
#                 feats = encoder(img1)
#                 logits = classifier(feats)
#                 loss = criterion(logits, labels)
#                 total += labels.size(0)
#                 correct += (logits.argmax(1) == labels).sum().item()
#                 total_loss += loss.item() * labels.size(0)
#         val_acc = correct / total
#         val_loss = total_loss / total
#         print(f"[Linear Eval][Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
#         writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(classifier.state_dict(), 'linear_classifier_best.pth')
#         print(f'6-3. Epoch {epoch+1} 종료')

# 6. (선택) Fine-tuning (Encoder Unfreeze, 전체/일부)
# encoder.train()
# for p in encoder.parameters():
#     p.requires_grad = True
# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
# ... 동일한 학습 루프 ...


# 7. Semi-supervised 실험: train_dataset 일부만 사용 (5%)
semi_size = int(len(train_dataset) * 0.05)
semi_train, _ = random_split(train_dataset, [semi_size, len(train_dataset)-semi_size])
semi_loader = DataLoader(semi_train, batch_size=64, shuffle=True)

# --- Semi-supervised Linear Evaluation ---
print(f"\n[SEMI-SUPERVISED] Training with only {semi_size} samples ({100*semi_size/len(train_dataset):.2f}%) of labeled data.")
classifier_semi = nn.Linear(128, num_classes).to(device)
for p in encoder.parameters():
    p.requires_grad = False
optimizer_semi = torch.optim.Adam(classifier_semi.parameters(), lr=1e-3)
criterion_semi = nn.CrossEntropyLoss()
num_epochs_semi = 20
best_acc_semi = 0

# --- CSV 기록용 ---
semi_logfile = 'semi_eval_log.csv'
with open(semi_logfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for epoch in range(num_epochs_semi):
        classifier_semi.train()
        total, correct, total_loss = 0, 0, 0
        for img1, img2, labels in semi_loader:
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
        train_acc = correct / total
        train_loss = total_loss / total
        # Validation
        classifier_semi.eval()
        total, correct, total_loss = 0, 0, 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, labels = img1.to(device), labels.to(device)
                feats = encoder(img1)
                logits = classifier_semi(feats)
                loss = criterion_semi(logits, labels)
                total += labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
        val_acc = correct / total
        val_loss = total_loss / total
        print(f"[Semi-supervised][Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        if val_acc > best_acc_semi:
            best_acc_semi = val_acc
            torch.save(classifier_semi.state_dict(), 'linear_classifier_semi_best.pth')

# 8. (선택) Supervised baseline: encoder+classifier 랜덤 초기화 후 전체 학습
# encoder = SimCLRVIT(out_dim=128).cuda()
# classifier = nn.Linear(128, num_classes).cuda()
# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
# ... 동일한 학습 루프 ...

print('실험 완료!')
