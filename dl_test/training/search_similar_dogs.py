import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F # 코사인 유사도 계산을 위해 임포트

# --- 프로젝트 구조에 맞춰 필요한 클래스/함수 임포트 ---
try:
    from training.extract_features import setup_feature_extractor 
except ImportError:
    try:
        from .extract_features import setup_feature_extractor 
    except ImportError:
        from extract_features import setup_feature_extractor 

# --- 설정 (extract_features.py 및 extract_db_features.py에서 가져옴) ---
OUT_DIM = 192 # 실제 DB 특징 벡터와 일치하도록 수정 (기존 128에서 변경)
MODEL_PATH = 'models/simclr_vit_dog_model.pth' 
IMAGE_SIZE = 224
DB_FEATURES_FILE = 'db_features.npy' # 저장된 DB 특징 파일
DB_IMAGE_PATHS_FILE = 'db_image_paths.npy' # 저장된 DB 이미지 경로 파일

def search_similar_dogs(
    query_image_path: str, 
    top_k: int = 5, 
    model_path: str = MODEL_PATH, 
    out_dim: int = OUT_DIM, 
    image_size: int = IMAGE_SIZE,
    db_features_file: str = DB_FEATURES_FILE,
    db_image_paths_file: str = DB_IMAGE_PATHS_FILE
) -> list:
    """
    사용자 쿼리 이미지와 가장 유사한 DB 내 강아지 이미지를 검색합니다.

    Args:
        query_image_path (str): 사용자가 업로드한 강아지 이미지 파일 경로.
        top_k (int): 가장 유사한 상위 K개의 결과를 반환합니다.
        model_path (str): 학습된 SimCLR 모델 가중치 파일 경로.
        out_dim (int): SimCLR 모델의 출력 임베딩 차원.
        image_size (int): 모델 훈련 시 사용한 이미지 크기.
        db_features_file (str): 저장된 DB 특징 벡터 NumPy 파일 경로.
        db_image_paths_file (str): 저장된 DB 이미지 경로 NumPy 파일 경로.

    Returns:
        list: (유사도 점수, 이미지 경로) 튜플의 리스트 (유사도 내림차순).
    """
    # 1. 특징 추출기 및 전처리 트랜스폼 설정
    # 이 부분은 한 번만 초기화되도록 전역적으로 관리하거나, 웹 서비스의 경우 서버 시작 시 로드되도록 해야 합니다.
    # 여기서는 함수 호출 시마다 로드되지만, 실제 서비스에서는 효율을 위해 최적화 필요.
    feature_extractor, preprocess_transform, device = setup_feature_extractor(
        model_path=model_path, out_dim=out_dim, image_size=image_size
    )

    # 2. DB 특징 벡터 및 이미지 경로 로드
    if not os.path.exists(db_features_file) or not os.path.exists(db_image_paths_file):
        raise FileNotFoundError(
            f"DB 특징 파일({db_features_file}) 또는 경로 파일({db_image_paths_file})을 찾을 수 없습니다.\n"
            "먼저 'extract_db_features.py'를 실행하여 DB 특징을 생성해야 합니다."
        )
    
    db_features = np.load(db_features_file)
    db_image_paths = np.load(db_image_paths_file)
    
    # NumPy 배열을 PyTorch 텐서로 변환하고 GPU로 이동 (한 번만 수행)
    db_features_tensor = torch.from_numpy(db_features).float().to(device)
    print(f"DB 특징 {db_features_tensor.shape} 로드 완료.")

    # 3. 쿼리 이미지 전처리 및 특징 추출
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"쿼리 이미지 파일이 없습니다: {query_image_path}")

    query_image = Image.open(query_image_path).convert('RGB')
    query_tensor = preprocess_transform(query_image)
    
    # 배치 차원 추가
    query_batch = query_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        query_features = feature_extractor(query_batch)
    
    # 특징 벡터를 정규화합니다. 코사인 유사도 계산 전에 정규화하는 것이 일반적입니다.
    # F.normalize는 기본적으로 L2 정규화를 수행합니다.
    query_features_normalized = F.normalize(query_features, p=2, dim=1)
    db_features_normalized = F.normalize(db_features_tensor, p=2, dim=1)
    
    # 4. 코사인 유사도 계산
    # (쿼리 특징 벡터 x DB 특징 벡터들)의 내적 = 코사인 유사도 (정규화된 벡터일 때)
    # query_features_normalized: [1, OUT_DIM]
    # db_features_normalized: [Num_DB_Images, OUT_DIM]
    # 결과 유사도: [1, Num_DB_Images]
    similarities = torch.matmul(query_features_normalized, db_features_normalized.transpose(0, 1))
    
    # 유사도 텐서를 NumPy 배열로 변환
    similarities_np = similarities.cpu().numpy().flatten() # 1차원 배열로 만듬

    # 5. 유사도 기준으로 상위 K개 결과 정렬
    # 유사도 점수가 높은 순서대로 인덱스를 가져옵니다.
    top_k_indices = similarities_np.argsort()[-top_k:][::-1] # 내림차순 정렬

    results = []
    for idx in top_k_indices:
        similarity_score = similarities_np[idx]
        image_path = db_image_paths[idx]
        results.append((similarity_score, image_path))
    
    return results

# --- 테스트 실행 예시 ---
if __name__ == "__main__":
    # 테스트를 위한 쿼리 이미지 경로 설정 (실제 강아지 이미지 경로로 변경해야 합니다)
    query_img_path = 'training/Images/n02085936-Maltese_dog/n02085936_10719.jpg' 
    
    # --- 만약 'data/stanford_dogs/Images' 경로가 없다면, 다른 이미지로 대체하거나 새로 다운로드해야 합니다. ---
    # 이 스크립트를 테스트하기 위해, 프로젝트에 임의의 강아지 이미지 파일 하나를 준비해 주세요.
    # 예를 들어, 'test_query_dog.jpg'라는 이름으로 src/ 에 저장하고 아래 경로를 사용합니다.
    # query_img_path = 'src/test_query_dog.jpg'

    if not os.path.exists(query_img_path):
        print(f"경고: 쿼리 이미지 '{query_img_path}'를 찾을 수 없습니다. 적절한 경로로 수정해 주세요.")
        # sys.exit(1) # sys 모듈 임포트 후 사용 가능
    
    print(f"쿼리 이미지: {query_img_path}")
    print(f"DB 특징 파일: {DB_FEATURES_FILE}, DB 경로 파일: {DB_IMAGE_PATHS_FILE}")

    try:
        top_similar_dogs = search_similar_dogs(query_image_path=query_img_path, top_k=5)
        
        print("\n--- 가장 유사한 강아지 검색 결과 (상위 5개) ---")
        for i, (score, path) in enumerate(top_similar_dogs):
            print(f"{i+1}. 유사도: {score:.4f}, 이미지 경로: {path}")
            

    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")