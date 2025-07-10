# SimCLR Loss 최적화 보고서: Loss < 1.0 → 0.32 달성 과정

## 📊 목표 및 달성 결과
- **목표**: SimCLR 파인튜닝에서 Loss < 1.0 달성
- **최종 결과**: Loss 1.35 → 0.32 (약 76% 감소)
- **달성 일자**: 2025년 7월 9일
- **사용 모델**: SimCLR + ViT (Vision Transformer)

---

## 🔧 핵심 변경사항 및 최적화

### 1. 데이터 로딩 시스템 개선 (`train_simclr_finetune.py`)

#### 1.1 Dataset 구조 변경
```python
# 변경 전: ImageFolder 사용 (폴더 구조 필요)
dataset = ImageFolder(DATA_DIR, transform=transforms)

# 변경 후: 커스텀 SimCLRDataset (단일 폴더 지원)
class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_ratio=1.0):
        self.image_paths = []
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(root_dir, file))
```

#### 1.2 DataLoader 최적화 (Windows 환경)
```python
# Windows 환경 최적화 설정
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=pil_collate_fn,     # 커스텀 collate 함수
    pin_memory=False,              # Windows 안정성
    num_workers=0,                 # Windows multiprocessing 문제 해결
    persistent_workers=False,      # 메모리 누수 방지
    drop_last=True                 # 배치 크기 일관성
)
```

#### 1.3 커스텀 Collate 함수 추가
```python
def pil_collate_fn(batch):
    """PIL 이미지 처리 전용 collate 함수"""
    try:
        images1, images2, labels = zip(*batch)
        images1 = torch.stack([img for img in images1 if img is not None])
        images2 = torch.stack([img for img in images2 if img is not None])
        labels = torch.tensor(labels)
        return images1, images2, labels
    except Exception as e:
        print(f"Collate 오류: {e}")
        return None, None, None
```

### 2. Mixed Precision 문제 해결

#### 2.1 오버플로우 문제 해결
```python
# 변경 전: Mixed Precision 사용 (오버플로우 발생)
with autocast():
    loss = nt_xent_loss(z1, z2, temperature=temperature)
scaler.scale(loss).backward()

# 변경 후: 일반 역전파 사용 (안정성 확보)
loss = nt_xent_loss(z1, z2, temperature=temperature)
loss.backward()
```

#### 2.2 NT-Xent Loss 마스크 값 조정
```python
# Mixed Precision 호환성을 위한 마스크 값 변경
# 변경 전: -9e15 (오버플로우 발생)
# 변경 후: -1e4 (안정적)
similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)
```

### 3. 하이퍼파라미터 최적화

#### 3.1 핵심 하이퍼파라미터 설정
```python
HYPERPARAMETERS = {
    "batch_size": 64,              # 메모리 효율성과 성능 균형
    "epochs": 80,                  # 충분한 학습 에포크
    "lr": 5e-5,                   # 작은 학습률로 안정적 학습
    "temperature": 0.1,            # NT-Xent Loss 온도 파라미터
    "weight_decay": 1e-4,          # L2 정규화
    "warmup_epochs": 5,            # 워밍업 에포크
    "gradient_clipping": 1.0,      # 그래디언트 클리핑
    "early_stopping": 10           # Early Stopping 인내심
}
```

#### 3.2 옵티마이저 변경
```python
# 변경 전: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 변경 후: AdamW (weight decay 내장)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 4. 학습률 스케줄러 개선

#### 4.1 워밍업 + 코사인 어닐링 스케줄러
```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 워밍업 단계: 선형 증가
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 코사인 어닐링 단계
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

### 5. 데이터 증강 강화

#### 5.1 강화된 데이터 증강 전략
```python
def get_enhanced_simclr_transforms(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        T.RandomRotation(degrees=15),           # 회전 추가
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 평행이동 추가
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

### 6. 학습 루프 최적화

#### 6.1 그래디언트 클리핑 추가
```python
# 그래디언트 폭발 방지
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
```

#### 6.2 Early Stopping 구현
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

#### 6.3 모델 저장 버전 관리
```python
def get_next_model_version(base_path):
    """모델 버전 자동 증가"""
    pattern = re.compile(r'.*_v(\d+)\.pth$')
    max_version = 0
    
    directory = os.path.dirname(base_path)
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
    
    return max_version + 1
```

### 7. 환경 및 디버깅 개선

#### 7.1 GPU 환경 체크 및 로깅
```python
def check_gpu_environment():
    """GPU 환경 정보 출력"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"Current Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
```

#### 7.2 tqdm 진행바 및 상세 로깅
```python
# 학습 진행 상황 시각화
from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, (img1, img2, _) in enumerate(progress_bar):
        # ... 학습 코드 ...
        
        # 진행바 업데이트
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}',
            'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
```

---

## 📈 실험 결과 추적 시스템

### 8. 실험 자동화 및 추적 (`experiment_tracker.py`)

#### 8.1 실험 설정 및 결과 자동 기록
```python
class ExperimentTracker:
    def log_experiment(self, config, results, experiment_name=None):
        """실험 설정과 결과를 JSON으로 저장"""
        experiment_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "config": config,
            "results": results
        }
```

#### 8.2 실험 결과 비교 및 시각화
```python
def compare_experiments(self):
    """모든 실험 결과 비교 분석 및 시각화"""
    # 배치 크기 vs Loss 산점도
    # 학습률 vs Loss 관계
    # 온도 파라미터 영향 분석
    # 목표 달성 현황 파이차트
```

---

## 🏆 주요 성과 및 학습 내용

### 성능 개선 결과
- **Loss 감소**: 1.35 → 0.32 (76% 개선)
- **목표 달성**: Loss < 1.0 조기 달성 (목표 대비 68% 추가 개선)
- **학습 안정성**: Early Stopping으로 과적합 방지
- **모델 품질**: v2 모델로 안정적 저장

### 기술적 학습 내용
1. **Mixed Precision의 한계**: 복잡한 Loss에서 오버플로우 이슈
2. **Windows 환경 최적화**: multiprocessing 제한과 해결책
3. **하이퍼파라미터 민감도**: 학습률과 온도 파라미터의 중요성
4. **데이터 증강 효과**: 강화된 증강이 일반화 성능 향상
5. **스케줄러 중요성**: 워밍업 + 코사인 어닐링의 효과

### 실전 워크플로우 구축
1. **자동화된 실험 추적**: JSON 기반 실험 로그
2. **버전 관리**: 모델 파일 자동 버전 증가
3. **시각화 시스템**: 실시간 성능 모니터링
4. **환경 최적화**: GPU 메모리 및 성능 최적화

---

## 🔍 문제 해결 과정

### 주요 문제들과 해결책

#### 1. Mixed Precision 오버플로우
- **문제**: `RuntimeError: value cannot be converted to type at::Half without overflow`
- **원인**: NT-Xent Loss의 큰 마스크 값 (-9e15)
- **해결**: 일반 역전파 사용 + 마스크 값 조정 (-1e4)

#### 2. Windows Multiprocessing 문제
- **문제**: DataLoader num_workers > 0 시 오류
- **해결**: num_workers=0, persistent_workers=False 설정

#### 3. 모델 저장 버전 관리
- **문제**: 수동 버전 관리의 불편함
- **해결**: 정규식 기반 자동 버전 증가 시스템

#### 4. 학습 불안정성
- **문제**: Loss 진동 및 발산
- **해결**: 그래디언트 클리핑 + Early Stopping + 워밍업 스케줄러

---

## 📁 변경된 파일 목록

### 핵심 파일들
1. **`train_simclr_finetune.py`** - 메인 파인튜닝 스크립트
2. **`experiment_tracker.py`** - 실험 자동화 시스템
3. **`performance.py`** - 성능 분석 및 시각화
4. **`linear_eval.py`** - Linear Evaluation (v2 모델 사용)

### 생성된 결과 파일들
- `simclr_vit_dog_model_finetuned_v2.pth` - 최종 모델
- `experiment_*.json` - 실험 로그 파일들
- `performance_plots/` - 다양한 성능 분석 차트들

---

## 🚀 향후 개선 방향

### 단기 목표
1. **실험 연동 자동화**: 학습 완료 시 자동으로 experiment_tracker에 결과 기록
2. **프론트엔드 연동**: 실시간 Loss 모니터링 대시보드
3. **추가 하이퍼파라미터 실험**: 온도, 증강 강도, 아키텍처 변형

### 장기 목표
1. **모델 해석성**: Attention 시각화, GradCAM 적용
2. **성능 최적화**: 추가 Loss 함수 실험 (InfoNCE 변형)
3. **실제 응용**: 개 품종 분류 웹 애플리케이션 완성

---

## 📚 참고 자료 및 코드 구조

### 프로젝트 구조
```
dl_test/
├── training/
│   ├── train_simclr_finetune.py    # 메인 파인튜닝
│   ├── experiment_tracker.py       # 실험 추적
│   ├── performance.py              # 성능 분석
│   ├── linear_eval.py              # Linear Evaluation
│   ├── model.py                    # SimCLR 모델 정의
│   └── dataset.py                  # Dataset 클래스
├── models/
│   └── simclr_vit_dog_model_finetuned_v2.pth
└── experiment_results/
    └── experiment_*.json
```

### 핵심 기술 스택
- **PyTorch**: 딥러닝 프레임워크
- **Vision Transformer (ViT)**: 백본 아키텍처
- **SimCLR**: Self-supervised Learning 방법론
- **Mixed Precision**: 메모리 최적화 (이후 제거)
- **tqdm**: 진행 상황 시각화
- **matplotlib/seaborn**: 결과 시각화

---

*이 문서는 SimCLR Loss 0.32 달성까지의 모든 기술적 변경사항과 학습 과정을 정리한 종합 보고서입니다.*

**최종 업데이트**: 2025년 7월 9일  
**작성자**: AI 개발 워크플로우 최적화 프로젝트
