import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import torch.nn.functional as F # 코사인 유사도 계산을 위해 임포트
import sys, os

# --- 프로젝트 구조에 맞춰 필요한 클래스/함수 임포트 ---
try:
    from training.extract_features import setup_feature_extractor 
except ImportError:
    try:
        from .extract_features import setup_feature_extractor 
    except ImportError:
        from extract_features import setup_feature_extractor 

# DB에서 이미지 정보 조회 함수 임포트
try:
    from database import get_dog_by_image_path
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
    try:
        from database import get_dog_by_image_path
    except ImportError:
        get_dog_by_image_path = None  # 테스트 환경 등에서 None 처리

# --- 설정 (extract_features.py 및 extract_db_features.py에서 가져옴) ---
OUT_DIM = 128
MODEL_PATH = 'models/simclr_vit_dog_model_finetuned_v1.pth'
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
    # --- 경로 디버깅용 출력 및 체크 추가 ---
    print(f"[SimCLR] 모델 경로(model_path): {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SimCLR 모델 가중치 파일이 없습니다: {model_path}\n경로를 확인하거나, 모델 파일을 올바른 위치에 복사하세요.")

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
    print(f"[DEBUG] db_features shape: {db_features.shape}, dtype: {db_features.dtype}, min: {db_features.min()}, max: {db_features.max()}")
    print(f"[DEBUG] db_image_paths shape: {db_image_paths.shape}, 예시: {db_image_paths[:3]}")
    # DB 벡터 값 분포 (샘플 5개)
    for i in range(min(5, db_features.shape[0])):
        print(f"[DEBUG] db_features[{i}] values: min={db_features[i].min()}, max={db_features[i].max()}, mean={db_features[i].mean()}, std={db_features[i].std()}")
    # NumPy 배열을 PyTorch 텐서로 변환하고 GPU로 이동 (한 번만 수행)
    db_features_tensor = torch.from_numpy(db_features).float().to(device)
    print(f"DB 특징 {db_features_tensor.shape} 로드 완료.")

    # 3. 쿼리 이미지 전처리 및 특징 추출
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"쿼리 이미지 파일이 없습니다: {query_image_path}")

    query_image = Image.open(query_image_path).convert('RGB')
    query_tensor = preprocess_transform(query_image)
    print(f"[DEBUG] query_tensor shape: {query_tensor.shape}, min: {query_tensor.min().item()}, max: {query_tensor.max().item()}")
    # 배치 차원 추가
    query_batch = query_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        query_features = feature_extractor(query_batch)
    print(f"[DEBUG] query_features shape: {query_features.shape}, min: {query_features.min().item()}, max: {query_features.max().item()}, mean: {query_features.mean().item()}, std: {query_features.std().item()}")
    # 특징 벡터를 정규화합니다. 코사인 유사도 계산 전에 정규화하는 것이 일반적입니다.
    # F.normalize는 기본적으로 L2 정규화를 수행합니다.
    query_features_normalized = F.normalize(query_features, p=2, dim=1)
    db_features_normalized = F.normalize(db_features_tensor, p=2, dim=1)
    print(f"[DEBUG] query_features_normalized norm: {torch.norm(query_features_normalized).item()}")
    print(f"[DEBUG] db_features_normalized norm (first 5): {[torch.norm(db_features_normalized[i]).item() for i in range(min(5, db_features_normalized.shape[0]))]}")
    # 4. 코사인 유사도 계산
    similarities = torch.matmul(query_features_normalized, db_features_normalized.transpose(0, 1))
    similarities_np = similarities.cpu().numpy().flatten() # 1차원 배열로 만듬
    print(f"[DEBUG] similarities_np (first 10): {similarities_np[:10]}")
    # 5. 유사도 기준으로 상위 K개 결과 정렬
    # 유사도 점수가 높은 순서대로 인덱스를 가져옵니다.
    top_k_indices = similarities_np.argsort()[-top_k:][::-1] # 내림차순 정렬

    results = []
    for idx in top_k_indices:
        similarity_score = similarities_np[idx]
        image_path = db_image_paths[idx]
        # 경로 포맷 통일 (DB 매칭 전)
        norm_image_path = image_path.replace("\\", "/")
        db_info = None
        image_url = None
        if get_dog_by_image_path:
            try:
                db_info = get_dog_by_image_path(norm_image_path)
                # image_url이 없으면 public_url, pet_image 등도 시도
                if db_info:
                    image_url = db_info.get('image_url')
                    if not image_url:
                        image_url = db_info.get('public_url') or db_info.get('pet_image')
                        if image_url:
                            db_info['image_url'] = image_url  # image_url 필드로 강제 추가
                    if not image_url:
                        print(f"[경고] DB에서 image_url을 찾을 수 없음: {norm_image_path} → {db_info}")
                else:
                    print(f"[경고] DB에서 이미지 정보 없음: {norm_image_path}")
            except Exception as e:
                print(f"[경고] DB 조회 실패: {e}")
                db_info = None
        else:
            print(f"[경고] get_dog_by_image_path 함수가 정의되어 있지 않음. DB 연동 불가.")
        results.append({
            'similarity': float(similarity_score),
            'image_path': image_path,
            'db_info': db_info
        })
    return results

def load_db_vectors_and_urls(
    db_host, db_user, db_password, db_name, db_port=3306
):
    """
    DB에서 모든 이미지의 벡터, URL을 불러온다.
    Returns:
        vectors: (N, OUT_DIM) numpy array
        urls:    (N,) list of str
    """
    import pymysql
    import json
    conn = pymysql.connect(
        host=db_host, user=db_user, password=db_password, db=db_name, port=db_port, charset='utf8mb4'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT public_url, image_vector FROM pet_image WHERE image_vector IS NOT NULL")
    rows = cursor.fetchall()
    urls, vectors = [], []
    for url, vec_json in rows:
        urls.append(url)
        try:
            vec = np.array(json.loads(vec_json))
            # print(f"[DEBUG] {url} image_vector 파싱 성공: shape={vec.shape}")  # 과도한 디버그 출력 주석 처리
        except Exception:
            vec = np.zeros(OUT_DIM)
        vectors.append(vec)
    conn.close()
    return np.stack(vectors), urls

def search_similar_dogs_db(
    query_image_path: str,
    db_host, db_user, db_password, db_name, db_port=3306,
    top_k: int = 5,
    model_path: str = MODEL_PATH,
    out_dim: int = OUT_DIM,
    image_size: int = IMAGE_SIZE
):
    """
    DB에서 벡터/URL을 직접 불러와 유사도 검색
    """
    feature_extractor, preprocess_transform, device = setup_feature_extractor(
        model_path=model_path, out_dim=out_dim, image_size=image_size
    )
    db_vectors, db_urls = load_db_vectors_and_urls(
        db_host, db_user, db_password, db_name, db_port
    )
    print(f"[DEBUG] DB 벡터 shape: {db_vectors.shape}, min: {db_vectors.min()}, max: {db_vectors.max()}")
    print(f"[DEBUG] DB URL 개수: {len(db_urls)}, 예시: {db_urls[:3]}")
    db_vectors_tensor = torch.from_numpy(db_vectors).float().to(device)
    # 쿼리 이미지 벡터 추출
    query_image = Image.open(query_image_path).convert('RGB')
    query_tensor = preprocess_transform(query_image).unsqueeze(0).to(device)
    print(f"[DEBUG] 쿼리 이미지 벡터 shape: {query_tensor.shape}, min: {query_tensor.min().item()}, max: {query_tensor.max().item()}")
    with torch.no_grad():
        query_features = feature_extractor(query_tensor)
    print(f"[DEBUG] 쿼리 특징 벡터 shape: {query_features.shape}, min: {query_features.min().item()}, max: {query_features.max().item()}")
    query_features_normalized = F.normalize(query_features, p=2, dim=1)
    db_features_normalized = F.normalize(db_vectors_tensor, p=2, dim=1)
    print(f"[DEBUG] query_features_normalized norm: {torch.norm(query_features_normalized).item()}")
    print(f"[DEBUG] db_features_normalized norm (first 5): {[torch.norm(db_features_normalized[i]).item() for i in range(min(5, db_features_normalized.shape[0]))]}")
    similarities = torch.matmul(query_features_normalized, db_features_normalized.transpose(0, 1))
    similarities_np = similarities.cpu().numpy().flatten()
    print(f"[DEBUG] similarities: {similarities_np[:10]}")
    top_k_indices = similarities_np.argsort()[-top_k:][::-1]
    print(f"[DEBUG] top_k 인덱스: {top_k_indices}")
    results = []
    for idx in top_k_indices:
        results.append({
            'similarity': float(similarities_np[idx]),
            'image_url': db_urls[idx],
        })
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
        print("\n[.npy 기반 검색 결과]")
        top_similar_dogs = search_similar_dogs(query_image_path=query_img_path, top_k=5)
        
        for i, result in enumerate(top_similar_dogs):
            score = result['similarity']
            path = result['image_path']
            db_info = result['db_info']
            print(f"{i+1}. 유사도: {score:.4f}, 이미지 경로: {path}, DB 정보: {db_info}")
            

    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")
    
    # 예시 DB 접속 정보 (실제 환경에 맞게 수정)
    db_host = 'byhou.synology.me'
    db_user = 'h3'
    db_password = 'Dbrlrus25^'
    db_name = 'h3'
    db_port = 3370

    print("\n[DB 기반 검색 결과]")
    try:
        top_similar_dogs_db = search_similar_dogs_db(
            query_image_path=query_img_path,
            db_host=db_host,
            db_user=db_user,
            db_password=db_password,
            db_name=db_name,
            db_port=db_port,
            top_k=5
        )
        for i, result in enumerate(top_similar_dogs_db):
            score = result['similarity']
            url = result['image_url']
            print(f"{i+1}. 유사도: {score:.4f}, 이미지 URL: {url}")
    except Exception as e:
        print(f"DB 기반 오류: {e}")