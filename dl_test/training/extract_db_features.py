import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image # Image 모듈 추가 임포트

# --- 프로젝트 구조에 맞춰 필요한 클래스/함수 임포트 ---
from extract_features import setup_feature_extractor 

# --- 설정 (train.py에서 가져옴, 필요에 따라 조정) ---
DATA_ROOT = 'training/Images' 
OUT_DIM = 128 
MODEL_PATH = 'models/simclr_vit_dog_model.pth' 
IMAGE_SIZE = 224
BATCH_SIZE = 128 
FEATURES_SAVE_FILE = 'db_features.npy'
IMAGE_PATHS_SAVE_FILE = 'db_image_paths.npy'

# --- 이곳으로 FeatureExtractionDataset 클래스를 옮겨주세요! ---
# 즉, extract_and_save_db_features 함수 정의보다 위에 위치해야 합니다.
class FeatureExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
        print(f"DB에 있는 총 {len(self.image_paths)}개의 이미지를 로드할 예정입니다.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path 
# --- FeatureExtractionDataset 클래스 이동 끝 ---


def extract_and_save_db_features(
    data_root=DATA_ROOT, 
    model_path=MODEL_PATH, 
    out_dim=OUT_DIM, 
    image_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE,
    features_save_file=FEATURES_SAVE_FILE,
    image_paths_save_file=IMAGE_PATHS_SAVE_FILE
):
    """
    DB 내 모든 강아지 이미지의 특징 벡터를 추출하고 NumPy 파일로 저장합니다.
    """
    feature_extractor, preprocess_transform, device = setup_feature_extractor(
        model_path=model_path, out_dim=out_dim, image_size=image_size
    )

    # FeatureExtractionDataset이 이제 글로벌 스코프에 있으므로 정상적으로 참조됩니다.
    db_dataset = FeatureExtractionDataset(root_dir=data_root, transform=preprocess_transform) 
    # num_workers는 멀티프로세싱을 사용하여 데이터 로딩을 가속화합니다.
    # Windows 환경에서 DataLoader의 num_workers를 사용할 때는
    # 이 스크립트의 실행 코드를 'if __name__ == "__main__":' 블록 안에 넣어야 합니다.
    # 이미 그렇게 되어 있으니 괜찮습니다.
    db_dataloader = DataLoader(db_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=os.cpu_count() // 2 or 1, pin_memory=True) # 최소 1로 설정

    all_features = []
    all_image_paths = []

    print("DB 이미지 특징 벡터 추출 시작...")
    with torch.no_grad(): 
        for images, paths in tqdm(db_dataloader, desc="Extracting DB Features"):
            images = images.to(device)
            features = feature_extractor(images)
            all_features.append(features.cpu().numpy())
            all_image_paths.extend(paths) 

    final_features_array = np.concatenate(all_features, axis=0)

    np.save(features_save_file, final_features_array)
    np.save(image_paths_save_file, np.array(all_image_paths)) 

    print(f"\n모든 DB 이미지의 특징 벡터 추출 및 저장 완료!")
    print(f"저장된 특징 벡터 파일: {features_save_file} (Shape: {final_features_array.shape})")
    print(f"저장된 이미지 경로 파일: {image_paths_save_file} (총 {len(all_image_paths)}개)")
    print(f"이제 이 파일을 불러와서 사용자 이미지와 유사도를 비교할 수 있습니다.")


if __name__ == "__main__":
    # Windows에서 DataLoader의 num_workers가 0보다 클 때 필요합니다.
    # 이 블록 안에 전체 실행 코드를 넣어두면 됩니다.
    extract_and_save_db_features()