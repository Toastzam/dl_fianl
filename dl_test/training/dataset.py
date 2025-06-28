import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random # random 모듈 임포트


class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_ratio=None): # sample_ratio 인자 추가
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 데이터 로드
        classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            class_dir = os.path.join(root_dir, class_name)
            
            # --- 데이터 샘플링 로직 추가 ---
            class_image_paths = []
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    class_image_paths.append(os.path.join(class_dir, img_name))
            
            if sample_ratio is not None:
                # 각 품종(클래스)별로 지정된 비율만큼 이미지를 랜덤 샘플링
                num_to_sample = int(len(class_image_paths) * sample_ratio)
                # 만약 품종별 이미지 수가 너무 적어 0장이 된다면 최소 1장이라도 샘플링
                if num_to_sample == 0 and len(class_image_paths) > 0:
                    num_to_sample = 1
                
                sampled_paths = random.sample(class_image_paths, num_to_sample)
            else:
                sampled_paths = class_image_paths
            
            for img_path in sampled_paths:
                self.image_paths.append(img_path)
                self.labels.append(idx)
            # --- 샘플링 로직 끝 ---

        print(f"총 {len(self.image_paths)}개의 이미지를 로드했습니다. (원본: {len(self.image_paths)}개에서 샘플링)") # 로드된 이미지 수 출력 수정

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)
        else:
            img1 = image
            img2 = image

        return img1, img2, label # SimCLR는 label을 직접 사용하지 않지만, 데이터셋 구조상 유지


# --- SimCLR 훈련을 위한 데이터 증강 파이프라인 ---
def get_simclr_transforms(image_size):
    """
    SimCLR 논문에서 제안하는 데이터 증강 파이프라인을 구현합니다.
    Args:
        image_size (int): 이미지 크기 (예: 224)
    Returns:
        torchvision.transforms.Compose: 두 개의 다른 증강을 적용하는 transform 객체
    """
    s = 1.0  # 색상 왜곡 강도 (SimCLR 논문 참고)
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    
    # SimCLR의 주요 증강 기법
    # RandomResizedCrop: 이미지 크기 조절 및 무작위 크롭
    # RandomHorizontalFlip: 무작위 수평 뒤집기
    # ColorJitter: 색상 왜곡
    # RandomGrayscale: 무작위 흑백 변환
    # GaussianBlur: 가우시안 블러
    # ToTensor: PIL 이미지를 PyTorch 텐서로 변환
    # Normalize: 이미지 픽셀 값을 정규화 (ImageNet 평균/표준편차 사용)
    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)), # 무작위 크롭 및 리사이즈
        transforms.RandomHorizontalFlip(), # 무작위 수평 뒤집기
        transforms.RandomApply([color_jitter], p=0.8), # 80% 확률로 색상 왜곡 적용
        transforms.RandomGrayscale(p=0.2), # 20% 확률로 흑백 변환
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//100*2+1)], p=0.5), # 50% 확률로 가우시안 블러 적용
        transforms.ToTensor(), # 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 통계로 정규화
    ])
    return simclr_transform

# --- 데이터셋 사용 예시 (이 부분은 나중에 train.py에 포함될 내용입니다) ---
if __name__ == "__main__":
    # 데이터셋의 'Images' 폴더 경로를 정확히 지정해주세요!
    # 예시: 'your_project_name/data/stanford_dogs/Images'
    # 실제 경로로 수정해야 합니다.
    data_root = 'training/images' # 이 경로를 실제 데이터셋 경로로 수정하세요!

    image_size = 224 # ViT 모델이 일반적으로 사용하는 이미지 크기

    # SimCLR 훈련을 위한 증강 파이프라인 생성
    simclr_transforms = get_simclr_transforms(image_size)

    # 데이터셋 인스턴스 생성
    try:
        dataset = StanfordDogsDataset(root_dir=data_root, transform=simclr_transforms)
        
        # DataLoader를 사용하여 데이터 배치로 불러오기
        # shuffle=True는 훈련 시 데이터를 섞어주는 역할을 합니다.
        # batch_size는 한 번에 모델에 입력할 이미지의 개수입니다.
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # 첫 번째 배치 데이터 확인
        print("\n데이터로더에서 첫 번째 배치 로드 중...")
        for batch_idx, (img_view1, img_view2) in enumerate(dataloader):
            print(f"배치 {batch_idx+1}:")
            print(f"  첫 번째 뷰 이미지 텐서 크기: {img_view1.shape}") # [batch_size, channels, height, width]
            print(f"  두 번째 뷰 이미지 텐서 크기: {img_view2.shape}")
            
            # 여기서 실제 이미지 데이터를 시각화하거나 추가 처리를 할 수 있습니다.
            # (지금은 크기만 확인)
            
            if batch_idx == 0: # 첫 번째 배치만 확인하고 종료
                break
        print("데이터 로딩 및 전처리 테스트 완료.")

    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("데이터셋 root_dir 경로를 정확히 설정했는지 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")