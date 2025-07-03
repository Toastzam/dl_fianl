import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- 프로젝트 구조에 맞춰 필요한 클래스 임포트 ---
# model.py에서 SimCLRVIT 모델을 불러옵니다.
try:
    from training.model import SimCLRVIT
except ImportError:
    try:
        from .model import SimCLRVIT
    except ImportError:
        from model import SimCLRVIT

# dataset.py에서 get_simclr_transforms 함수를 불러옵니다.
try:
    from training.dataset import get_simclr_transforms
except ImportError:
    try:
        from .dataset import get_simclr_transforms
    except ImportError:
        from dataset import get_simclr_transforms

# --- 설정 (train.py에서 가져옴) ---
# SimCLR 모델의 출력 임베딩 차원 (train.py에서 OUT_DIM과 동일하게 설정)
OUT_DIM = 128 
# 모델 가중치가 저장된 경로 (train.py의 SAVE_PATH와 동일하게 설정)
SAVE_PATH = 'models/simclr_vit_dog_model_finetuned_v1.pth'
# 모델 훈련 시 사용한 이미지 크기 (train.py의 IMAGE_SIZE와 동일하게 설정)
IMAGE_SIZE = 224

# --- 특징 추출기 준비 함수 ---
def setup_feature_extractor(model_path=SAVE_PATH, out_dim=OUT_DIM, image_size=IMAGE_SIZE):
    """
    학습된 SimCLR 모델을 불러와 특징 추출기를 설정합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"특징 추출에 사용할 장치: {device}")

    # 1. SimCLR 모델 인스턴스 생성 (학습 시 사용했던 것과 동일한 구조)
    # model.py에서 vit_tiny_patch16_224를 사용하셨으므로, 그 구조 그대로 모델을 만듭니다.
    model = SimCLRVIT(out_dim=out_dim)
    model.to(device)

    # 2. 학습된 가중치 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {model_path}\n훈련을 먼저 완료해주세요.")
    
    # 모델 가중치를 로드합니다.
    # 이때 map_location을 사용하여 현재 사용 가능한 장치로 가중치를 로드할 수 있습니다.
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"모델 가중치 로드 완료: {model_path}")

    # 3. 특징 추출기는 전체 모델 사용 (백본 + projection_head)
    # SimCLR 모델 전체를 사용해야 128차원 출력을 얻을 수 있습니다.
    # backbone만 사용하면 ViT tiny의 원본 차원(192)이 나옵니다.
    feature_extractor = model  # 전체 모델 사용 
    
    # 4. 모델을 평가 모드로 설정 (중요!)
    # 드롭아웃, 배치 정규화 등이 평가 모드에 맞게 작동하도록 설정합니다.
    feature_extractor.eval() 
    print("특징 추출기 준비 완료.")

    # 5. 이미지 전처리 트랜스폼 설정 (학습 시 사용한 것과 동일해야 함)
    # SimCLR 학습 시 사용했던 정규화 및 크기 조절 등을 그대로 적용합니다.
    # 데이터셋에서 사용한 get_simclr_transforms 함수를 재활용합니다.
    # 하지만 특징 추출 시에는 이미지를 2개로 증강할 필요가 없으므로,
    # 단순히 하나의 이미지에 적용되는 표준 전처리 트랜스폼을 정의합니다.
    preprocess_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return feature_extractor, preprocess_transform, device

# --- 특징 추출 테스트 예시 ---
if __name__ == "__main__":
    # 특징 추출기 및 전처리 트랜스폼 설정
    feature_extractor, preprocess_transform, device = setup_feature_extractor()

    # 테스트를 위한 샘플 이미지 경로
    sample_image_path = 'training/Images/n02085936-Maltese_dog/n02085936_10719.jpg' 
    
    if not os.path.exists(sample_image_path):
        print(f"경고: 샘플 이미지 경로 '{sample_image_path}'를 찾을 수 없습니다. 테스트를 건너뜜.")
        print("실제 강아지 이미지 경로로 'sample_image_path'를 수정하여 다시 시도하세요.")
    else:
        # 샘플 이미지 로드 및 전처리
        sample_image = Image.open(sample_image_path).convert('RGB')
        input_tensor = preprocess_transform(sample_image)
        
        # 배치 차원 추가 (모델은 배치 입력을 기대합니다)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # 특징 추출
        with torch.no_grad(): # 그라디언트 계산 비활성화 (메모리 절약 및 속도 향상)
            features = feature_extractor(input_batch)
        
        print(f"추출된 특징 벡터의 모양 (shape): {features.shape}")
        # 예상 출력: torch.Size([1, 768]) (ViT-Tiny의 출력 차원은 768입니다)
        print(f"추출된 특징 벡터 (일부): {features[0, :5].cpu().numpy()}") # 상위 5개 값 출력

    print("\n특징 추출기 테스트 완료.")
    print("이제 이 feature_extractor를 사용하여 DB의 모든 이미지와 사용자 업로드 이미지를 벡터화할 수 있습니다.")