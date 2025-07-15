import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
from database import get_all_dogs, get_dog_by_id, add_dog, update_dog, delete_dog, get_dog_by_image_path, add_image_mapping, DatabaseManager, get_breed_codes, get_breed_name_by_code
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys
import tempfile
import shutil
from PIL import Image
import random

from feature_extraction_service import get_feature_service
# retrain_manager에서 retraining 트리거 함수 임포트
from retrain_manager import trigger_retraining_if_needed

# Pydantic 모델 정의
class ImageUrlRequest(BaseModel):
    image_url: str

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 실제 모델 임포트 (에러 처리 포함)
ap10k_model = None
device = None
visualizer = None
search_similar_dogs = None

# 특징 추출 서비스 인스턴스
feature_service = get_feature_service()

try:
    import training.visualize_keypoints as vk
    import training.search_similar_dogs as ssd
    
    setup_ap10k_model = vk.setup_ap10k_model
    detect_and_visualize_keypoints = vk.detect_and_visualize_keypoints
    calculate_keypoint_similarity = vk.calculate_keypoint_similarity
    search_similar_dogs = ssd.search_similar_dogs
    
    MODELS_AVAILABLE = True
    print("✅ 모델 모듈 임포트 성공")
except ImportError as e:
    print(f"⚠️ 모델 모듈 임포트 실패: {e}")
    print("🔄 더미 모드로 실행됩니다")
    MODELS_AVAILABLE = False

# FastAPI 앱 초기화 (모델 로드 포함)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 모델 로드
    global ap10k_model, device, visualizer
    
    if MODELS_AVAILABLE:
        try:
            print("🚀 AP-10K 모델 로딩 중...")
            ap10k_model, device, visualizer = setup_ap10k_model()
            print("✅ AP-10K 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🔄 더미 모드로 계속 실행됩니다")
    else:
        print("🔄 모델 모듈이 없어 더미 모드로 실행됩니다")
    
    yield
    
    # 서버 종료 시 정리
    print("🔄 서버 종료 중...")

# FastAPI 앱 (lifespan 포함)

# redirect_slashes=False 옵션 추가로 307 리다이렉트 방지
app = FastAPI(
    title="Dog Similarity Search API", 
    description="SimCLR + AP-10K 키포인트 기반 강아지 유사도 검색",
    lifespan=lifespan,
    redirect_slashes=False
)


# 업로드/출력 폴더 절대경로로 고정 (output_keypoints는 반드시 프로젝트 루트)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output_keypoints")
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.0.46:3000",  # 다른 로컬에서 접속 가능하도록 추가
        "http://192.168.0.*:3000",
        "http://localhost:5173",
        "*"  # 모든 도메인 허용 (개발용, 운영시에는 구체적 IP 설정)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# SimCLR/Keypoint 유사도 가중치 상수 (합이 1.0이 되도록 조정)
SIMCLR_WEIGHT = 0.9
KEYPOINT_WEIGHT = 0.1

# SimCLR 모델 파일명(버전) 자동 추출
def get_simclr_model_version(path):
    # 경로에서 파일명만 추출 (ex: simclr_vit_dog_model_finetuned_v1.pth)
    fname = os.path.basename(path)
    # 확장자 제거
    if fname.endswith('.pth'):
        fname = fname[:-4]
    return fname

SIMCLR_MODEL_VERSION = get_simclr_model_version(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'simclr_vit_dog_model_finetuned_v2.pth')
)

# SimCLR 관련 설정 (항상 절대경로 사용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend 기준으로 절대경로
SIMCLR_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'simclr_vit_dog_model_finetuned_v2.pth')
print(f"[DEBUG] SIMCLR_MODEL_PATH 1: {os.path.abspath(SIMCLR_MODEL_PATH)}, exists: {os.path.exists(SIMCLR_MODEL_PATH)}")
if not os.path.exists(SIMCLR_MODEL_PATH):
    # fallback: 프로젝트 루트의 models/ 경로 시도
    alt_path = os.path.join(BASE_DIR, '..', '..', 'models', 'simclr_vit_dog_model_finetuned_v2.pth')
    print(f"[DEBUG] SIMCLR_MODEL_PATH 2 (alt): {os.path.abspath(alt_path)}, exists: {os.path.exists(alt_path)}")
    if os.path.exists(alt_path):
        SIMCLR_MODEL_PATH = alt_path
    else:
        print(f"[경고] SimCLR 모델 파일을 찾을 수 없습니다: {SIMCLR_MODEL_PATH} 또는 {alt_path}")

SIMCLR_OUT_DIM = 128  # 실제 저장된 모델과 일치하도록 복원
SIMCLR_IMAGE_SIZE = 224
# 더 이상 .npy 파일을 사용하지 않음 - DB에서 직접 벡터 로드





# Static files 마운트 - output_keypoints는 반드시 프로젝트 루트 output_keypoints
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/training", StaticFiles(directory=os.path.abspath(os.path.join(BASE_DIR, "..", "training"))), name="training")

@app.post("/api/upload_and_search/")
async def upload_and_search(file: UploadFile = File(...)):
    """실제 업로드 및 유사도 검색 API"""
    try:
        # 파일명 정규화 함수 임포트
        try:
            from training.visualize_keypoints import normalize_filename
        except ImportError:
            def normalize_filename(filename):
                import re
                name, ext = os.path.splitext(filename)
                name = re.sub(r'[^\w\d가-힣]+', '_', name)
                name = re.sub(r'_+', '_', name)
                name = name.strip('_')
                return name + ext


        # 파일명 유효성 체크 및 정규화
        if not file.filename or not file.filename.strip():
            raise HTTPException(status_code=400, detail="업로드 파일명이 비어 있습니다.")
        normalized_filename = normalize_filename(file.filename)
        if not normalized_filename or normalized_filename in ['.', '..', '']:
            raise HTTPException(status_code=400, detail="정규화된 파일명이 비어 있습니다.")
        file_location = os.path.join(UPLOAD_FOLDER, normalized_filename)
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"📁 파일 저장 완료: {file_location}")

        # 디버깅 정보 출력
        print(f"🔍 MODELS_AVAILABLE: {MODELS_AVAILABLE}")
        print(f"🔍 ap10k_model is not None: {ap10k_model is not None}")
        print(f"🔍 조건 체크: {MODELS_AVAILABLE and ap10k_model is not None}")

        if MODELS_AVAILABLE and ap10k_model is not None:
            # 실제 모델 사용
            print("🚀 실제 모델 모드로 진입")
            return await real_model_search(file_location, normalized_filename)
        else:
            # 더미 모드
            print("🔄 더미 모드로 진입")
            return await dummy_search(file_location, normalized_filename)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def real_model_search(file_location: str, filename: str):
    """실제 모델을 사용한 검색 (DB public_url 기반만 사용)"""
    import time
    start_time = time.time()
    try:
        print("🔍 실제 모델 사용 - 시작 (DB public_url 기반)")
        # 1. 쿼리 이미지 키포인트 검출
        print("🔍 단계 1: 쿼리 이미지 키포인트 검출")
        # output_path 인자가 없는 버전: 반환된 경로를 그대로 사용
        query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
            file_location, ap10k_model, device, visualizer
        )
        print(f"✅ 쿼리 이미지 키포인트 검출 완료: {query_kp_output_path}")
        # print(f"[DEBUG] query_pose_results: {query_pose_results}")

        # --- 강아지 판별 휴리스틱 1: 키포인트 개수/신뢰도 ---
        # query_pose_results: [{'keypoints': np.ndarray, ...}, ...] 또는 dict 형식 예상
        import numpy as np
        if isinstance(query_pose_results, list) and len(query_pose_results) > 0 and isinstance(query_pose_results[0], dict):
            kp_data = query_pose_results[0]
        elif isinstance(query_pose_results, dict):
            kp_data = query_pose_results
        else:
            kp_data = {}

        keypoints = kp_data.get('keypoints', [])
        # keypoints가 numpy array이고 shape가 (N, 3)일 때 score 추출
        if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 3:
            scores = keypoints[:, 2]
        else:
            scores = []
        num_keypoints = len(keypoints) if isinstance(keypoints, np.ndarray) else 0
        avg_score = float(np.mean(scores)) if isinstance(scores, np.ndarray) and scores.size > 0 else 0.0
        print(f"[DOG CHECK] keypoints: {num_keypoints}, avg_score: {avg_score:.3f}")
        # 실제 판별: 키포인트 개수와 평균 신뢰도 기준
        is_dog_by_kp = (num_keypoints >= 10 and avg_score >= 0.25)

        # 2. SimCLR 기반 유사 이미지 검색 (DB public_url만 사용)
        print("🔍 단계 2: SimCLR 기반 유사 이미지 검색 (DB public_url만)")
        from database import get_all_pet_images  # pet_image 테이블 전체 불러오기 (public_url, image_vector)
        pet_images = get_all_pet_images()  # [{'public_url': ..., 'image_vector': ...}, ...]
        print(f"[DEBUG] get_all_pet_images() 반환: {len(pet_images)}개")
        if pet_images:
            pass  # print(f"[DEBUG] pet_images[0] 예시: ...")  # [자동 주석처리] pet_images[0] 예시 출력
        if not pet_images:
            raise Exception("DB에 등록된 강아지 이미지가 없습니다.")
        # 쿼리 이미지 벡터 추출
        print(f"[DEBUG] 쿼리 이미지 벡터 추출 시작: {file_location}")
        query_vector = feature_service.extract_features_from_path(file_location)
        print(f"[DEBUG] 쿼리 이미지 벡터 shape: {query_vector.shape if hasattr(query_vector, 'shape') else type(query_vector)}")
        # 모든 DB 이미지와 유사도 계산
        import numpy as np
        db_vectors = np.stack([img['image_vector'] for img in pet_images])
        print(f"[DEBUG] DB 벡터 shape: {db_vectors.shape}")
        similarities = np.dot(db_vectors, query_vector) / (np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-8)
        print(f"[DEBUG] similarities: {similarities}")
        # --- 강아지 판별 휴리스틱 2: SimCLR DB 유사도 ---
        max_sim = float(np.max(similarities)) if similarities is not None and len(similarities) > 0 else 0.0
        print(f"[DOG CHECK] SimCLR max similarity to DB: {max_sim:.3f}")
        SIMCLR_MIN_SIM = 0.28  # 실험적으로 조정 (0.3 미만이면 강아지 아닐 확률 높음)
        is_dog_by_simclr = (max_sim >= SIMCLR_MIN_SIM)

        # --- 최종 강아지 판별 ---
        if not (is_dog_by_kp and is_dog_by_simclr):
            reason = []
            if not is_dog_by_kp:
                reason.append(f"키포인트 개수/신뢰도 부족 (개수: {num_keypoints}, 평균: {avg_score:.2f})")
            if not is_dog_by_simclr:
                reason.append(f"SimCLR DB 유사도 낮음 (최대: {max_sim:.2f})")
            msg = "업로드된 이미지는 강아지로 인식되지 않습니다. " + ", ".join(reason)
            print(f"[DOG CHECK] ❌ {msg}")
            def to_output_keypoints_url(path):
                if not path:
                    return None
                return f"/output_keypoints/{os.path.basename(path)}"
            return JSONResponse({
                'success': False,
                'error': 'not_a_dog',
                'message': msg,
                'query_image': file_location.replace('\\', '/'),
                'query_keypoint_image': to_output_keypoints_url(query_kp_output_path) if query_kp_output_path else None,
                'dog_check': {
                    'is_dog_by_kp': is_dog_by_kp,
                    'is_dog_by_simclr': is_dog_by_simclr,
                    'num_keypoints': num_keypoints,
                    'avg_score': avg_score,
                    'max_simclr_similarity': max_sim
                }
            }, status_code=400)

        # --- 이하 기존 유사도 검색 로직 (강아지로 판별된 경우만) ---
        # top_k 추출
        top_k = 6
        # 유사도 값이 nan이거나 inf인 경우 제외
        valid_indices = [i for i, s in enumerate(similarities) if np.isfinite(s)]
        similarities_valid = similarities[valid_indices]
        if len(similarities_valid) < top_k:
            print(f"[경고] 유효한 유사도 결과가 {len(similarities_valid)}개만 존재합니다. (top_k={top_k})")
        # 내림차순 정렬 후 상위 top_k 인덱스 추출
        sorted_indices = np.argsort(similarities_valid)[::-1][:top_k]
        top_indices = [valid_indices[i] for i in sorted_indices]
        print(f"[DEBUG] top_k 인덱스: {top_indices}")
        similar_results = []
        for idx in top_indices:
            simclr_score = float(similarities[idx])
            db_img = pet_images[idx].copy()
            if 'image_vector' in db_img:
                del db_img['image_vector']
            db_image_url = db_img.get('public_url') or db_img.get('image_url') or db_img.get('image_path')
            if db_image_url:
                base_name = os.path.basename(str(db_image_url))
                # 외부 URL(http/https)은 필터링하지 않고 그대로 사용
                if str(db_image_url).startswith('http://') or str(db_image_url).startswith('https://'):
                    pass
                else:
                    uploads_path = os.path.join(UPLOAD_FOLDER, base_name)
                    if os.path.exists(uploads_path):
                        db_image_url = f"/uploads/{base_name}"
                    else:
                        # uploads 폴더에 없고, 외부 URL도 아니면 기존 경로 그대로 사용 (예: static, output_keypoints 등)
                        pass
            else:
                db_image_url = None
            similar_results.append({
                'similarity': simclr_score,
                'image_url': db_image_url,
                'db_info': db_img
            })
        print(f"✅ SimCLR(DB) 검색 완료: {len(similar_results)}개 결과 (경로/URL 필터링 개선)")

        # 3. 각 유사 이미지에 대해 키포인트 검출 및 유사도 계산 (public_url만)
        print("🔍 단계 3: 유사 이미지들의 키포인트 검출 및 종합 유사도 계산 (public_url)")
        results = []
        import time
        for i, sim_result in enumerate(similar_results):
            simclr_score = sim_result.get('similarity', 0.0)
            image_url = sim_result.get('image_url')
            db_info = sim_result.get('db_info', {}).copy()
            if 'image_vector' in db_info:
                del db_info['image_vector']
            print(f"  🔍 유사 이미지 {i+1}/{len(similar_results)} 처리: {os.path.basename(image_url)}")
            keypoint_similarity = 0.0
            similar_kp_output_path = None
            keypoint_time = None
            try:
                kp_start = time.time()
                similar_kp_output_path, similar_pose_results = detect_and_visualize_keypoints(
                    image_url, ap10k_model, device, visualizer
                )
                if query_pose_results and similar_pose_results:
                    keypoint_similarity = calculate_keypoint_similarity(
                        query_pose_results, similar_pose_results
                    )
                kp_end = time.time()
                keypoint_time = round(kp_end - kp_start, 3)
            except Exception as e:
                print(f"⚠️ 키포인트 검출 실패 (URL: {image_url}): {e}")
                keypoint_time = None
            combined_similarity = (SIMCLR_WEIGHT * simclr_score) + (KEYPOINT_WEIGHT * keypoint_similarity)
            result_dict = {
                'rank': i + 1,
                'image_url': image_url,
                'keypoint_image_path': similar_kp_output_path.replace('\\', '/') if similar_kp_output_path else None,
                'simclr_similarity': float(simclr_score),
                'keypoint_similarity': float(keypoint_similarity),
                'combined_similarity': float(combined_similarity),
                'db_info': db_info,
                'keypoint_processing_time': keypoint_time
            }
            results.append(result_dict)
            print(f"    ✅ SimCLR: {simclr_score:.4f}, 키포인트: {keypoint_similarity:.4f}, 복합: {combined_similarity:.4f}, 키포인트시간: {keypoint_time}s")
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        print("🔍 단계 4: 복합 유사도로 재정렬 완료")
        for i, result in enumerate(results):
            result['rank'] = i + 1
        try:
            from database import get_all_dogs
            total_dogs = len(get_all_dogs())
        except:
            total_dogs = 10000
        processing_time = time.time() - start_time
        print(f"[DEBUG] 전체 처리 시간: {processing_time:.2f}초")
        def to_output_keypoints_url(path):
            if not path:
                return None
            # output_keypoints 폴더 내 파일명만 추출
            return f"/output_keypoints/{os.path.basename(path)}"
        # results 내부 keypoint_image_path도 URL로 변환
        for r in results:
            if r.get('keypoint_image_path'):
                r['keypoint_image_path'] = to_output_keypoints_url(r['keypoint_image_path'])
        # 쿼리 keypoint 이미지도 반환 경로에서 파일명만 추출해서 URL로 반환
        query_keypoint_url = to_output_keypoints_url(query_kp_output_path)
        # 전체 처리시간은 search_metadata에, 각 결과별 키포인트 처리시간은 results에 이미 포함됨
        # 모델명은 SIMCLR_MODEL_VERSION에서 추출, 필요시 최신값으로 재설정
        # 모델 버전 추출 함수 재호출 (혹시 모델 경로가 동적으로 바뀌는 경우)
        model_version = get_simclr_model_version(SIMCLR_MODEL_PATH)
        return JSONResponse({
            'success': True,
            'query_image': '/uploads/' + os.path.basename(file_location),
            'query_keypoint_image': query_keypoint_url,
            'results': results,
            'mode': 'real_model_db_public_url',
            'search_metadata': {
                'database_size': total_dogs,
                'images_with_data': total_dogs,
                'searched_results': len(results),
                'confidence_threshold': 0.60,
                'algorithm': 'SimCLR + AP-10K Hybrid AI',
                'processing_time': round(processing_time, 2),
                'model_version': model_version,
                'feature_dimension': 2048
            }
        })
    except Exception as e:
        print(f"❌ 실제 모델 검색 중 오류: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 더미 모드로 폴백")
        return await dummy_search(file_location, filename)

async def dummy_search(file_location: str, filename: str):
    """더미 데이터를 사용한 검색 (테스트/폴백용) - 실제 DB 이미지 사용"""
    print("🔄 더미 모드 사용 - 실제 DB에서 랜덤 샘플링")
    
    import time
    start_time = time.time()
    
    try:
        # DB 통계 정보만 빠르게 가져오기
        from database import get_all_dogs
        print("📊 DB 기본 통계 조회 중...")
        # 전체 데이터 수만 빠르게 계산
        total_dogs = len(get_all_dogs())
        print(f"📊 전체 강아지 수: {total_dogs}마리")

        # 만약 DB가 비어있으면 즉시 완전 폴백 더미 데이터로 반환 (6개)
        if total_dogs == 0:
            print("⚠️ DB가 비어있음: 완전 폴백 더미 데이터 사용 (6개)")
            fallback_results = [
                {
                    'rank': i + 1,
                    'id': i + 1,
                    'name': f'더미 강아지 {i + 1}',
                    'breed': ['골든 리트리버', '래브라도', '비글', '포메라니안', '믹스견', '푸들'][i % 6],
                    'breed_code': f'BREED_00{i+1}',
                    'gender': 'M' if i % 2 == 0 else 'F',
                    'weight': 15.0 + i * 2.5,
                    'color': ['갈색', '검은색', '흰색', '크림색', '회색', '브라운'][i % 6],
                    'description': f'더미 모드 테스트 강아지 {i + 1}',
                    'location': '서울시 강남구',
                    'adoption_status': 'APPLY_AVAILABLE',
                    'image_url': None,
                    'image_path': f'sample_dog_{i + 1}.jpg',
                    'keypoint_image_path': None,
                    'simclr_similarity': 0.85 - i * 0.05,
                    'keypoint_similarity': 0.75 - i * 0.03,
                    'combined_similarity': 0.82 - i * 0.04,
                    'similarity': 0.85 - i * 0.05,
                    'overall_similarity': 0.82 - i * 0.04
                }
                for i in range(6)
            ]
            return JSONResponse({
                'success': True,
                'query_image': file_location.replace('\\', '/'),
                'query_keypoint_image': None,
                'results': fallback_results,
                'mode': 'fallback_dummy',
                'search_metadata': {
                    'database_size': 0,
                    'images_with_data': 0,
                    'searched_results': len(fallback_results),
                    'confidence_threshold': 0.60,
                    'algorithm': 'Fallback Dummy Mode',
                    'processing_time': 0.1,
                    'model_version': 'fallback',
                    'feature_dimension': 128
                }
            })

        # 랜덤 검색용 샘플만 소량 가져오기
        print("🎲 검색용 샘플 데이터 생성 중...")
        # 더미 결과 직접 생성 (DB 조회 최소화)
        dummy_results = []
        for i in range(6):
            # 가상의 강아지 ID (실제 범위 내에서)
            fake_id = random.randint(1, min(total_dogs, 1000))
            # 더미 유사도 점수
            simclr_sim = random.uniform(0.7, 0.95)
            keypoint_sim = random.uniform(0.6, 0.9)
            combined_sim = (SIMCLR_WEIGHT * simclr_sim) + (KEYPOINT_WEIGHT * keypoint_sim)
            dummy_results.append({
                'rank': i + 1,
                'id': fake_id,
                'name': f'강아지 #{fake_id}',
                'breed': random.choice(['믹스견', '시바견', '푸들', '말티즈', '포메라니안', '푸들']),
                'breed_code': random.choice(['307', '208', '156', '178', '213', '999']),
                'gender': random.choice(['M', 'F', 'Q']),
                'gender_code': random.choice(['M', 'F', 'Q']),
                'weight': round(random.uniform(2.0, 25.0), 1),
                'color': random.choice(['갈색', '흰색', '검은색', '믹스', '크림색', '브라운']),
                'description': '더미 모드 테스트 강아지',
                'location': '서울시 강남구',
                'adoption_status': random.choice(['PREPARING', 'APPLY_AVAILABLE']),
                'adoption_status_code': random.choice(['PREPARING', 'APPLY_AVAILABLE']),
                'image_url': f'http://example.com/dog_{fake_id}.jpg',
                'image_path': f'http://example.com/dog_{fake_id}.jpg',
                'keypoint_image_path': None,
                'simclr_similarity': float(simclr_sim),
                'keypoint_similarity': float(keypoint_sim),
                'combined_similarity': float(combined_sim),
                'similarity': float(simclr_sim),
                'overall_similarity': float(combined_sim)
            })
        # 복합 유사도로 정렬
        dummy_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        # 순위 업데이트
        for i, result in enumerate(dummy_results):
            result['rank'] = i + 1
        print(f"✅ 고속 더미 모드 완료: {len(dummy_results)}마리 생성")
        # 각 강아지의 정보 출력
        print("\n🖼️  검색 결과 정보:")
        for i, dog in enumerate(dummy_results):
            print(f"  {i+1}. 강아지 ID: {dog.get('id')}")
            print(f"      이름: {dog.get('name')}")
            print(f"      견종: {dog.get('breed')} (코드: {dog.get('breed_code')})")
            print(f"      성별 코드: {dog.get('gender_code')} (프론트에서 변환됨)")
            print(f"      입양상태 코드: {dog.get('adoption_status_code')} (프론트에서 변환됨)")
            print(f"      유사도: {dog.get('combined_similarity', 0):.3f}")
            print()
        # 처리 시간 계산
        processing_time = time.time() - start_time
        # 검색 메타데이터 생성 (고속 추정값 사용)
        search_metadata = {
            'database_size': total_dogs,
            'images_with_data': int(total_dogs * 0.95),  # 95% 추정
            'searched_results': len(dummy_results),
            'confidence_threshold': 0.60,
            'algorithm': 'SimCLR + AP-10K Keypoints (Fast Dummy)',
            'processing_time': round(processing_time, 2),
            'feature_dimension': 2048,
            'model_version': SIMCLR_MODEL_VERSION + '-dummy'
        }
        return JSONResponse({
            'success': True,
            'query_image': file_location.replace('\\', '/'),
            'query_keypoint_image': None,
            'results': dummy_results,
            'mode': 'dummy_with_real_db',
            'search_metadata': search_metadata
        })
    except Exception as e:
        print(f"❌ 더미 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        # 완전 폴백: 하드코딩된 더미 데이터 (6개, 랜덤 사용 금지)
        fallback_results = [
            {
                'rank': i + 1,
                'id': i + 1,
                'name': f'더미 강아지 {i + 1}',
                'breed': ['골든 리트리버', '래브라도', '비글', '포메라니안', '믹스견', '푸들'][i % 6],
                'breed_code': f'BREED_00{i+1}',
                'gender': 'M' if i % 2 == 0 else 'F',
                'weight': 15.0 + i * 2.5,
                'color': ['갈색', '검은색', '흰색', '크림색', '회색', '브라운'][i % 6],
                'description': f'더미 모드 테스트 강아지 {i + 1}',
                'location': '서울시 강남구',
                'adoption_status': 'APPLY_AVAILABLE',
                'image_url': None,
                'image_path': f'sample_dog_{i + 1}.jpg',
                'keypoint_image_path': None,
                'simclr_similarity': 0.85 - i * 0.05,
                'keypoint_similarity': 0.75 - i * 0.03,
                'combined_similarity': 0.82 - i * 0.04,
                'similarity': 0.85 - i * 0.05,
                'overall_similarity': 0.82 - i * 0.04
            }
            for i in range(6)
        ]
        print("🔄 완전 폴백 더미 데이터 사용 (예외, 랜덤 없음)")
        return JSONResponse({
            'success': True,
            'query_image': file_location.replace('\\', '/'),
            'query_keypoint_image': None,
            'results': fallback_results,
            'mode': 'fallback_dummy',
            'search_metadata': {
                'database_size': 0,
                'images_with_data': 0,
                'searched_results': len(fallback_results),
                'confidence_threshold': 0.60,
                'algorithm': 'Fallback Dummy Mode',
                'processing_time': 0.1,
                'model_version': SIMCLR_MODEL_VERSION + '-fallback',
                'feature_dimension': 128
            }
        })

@app.post("/api/extract_features/")
async def extract_features_api(file: UploadFile = File(...)):
    """이미지에서 특징 벡터만 추출해서 반환 (등록 시스템용 API)"""
    try:
        # 파일 내용 읽기
        file_content = await file.read()
        
        print(f"📁 벡터 추출 요청: {file.filename}")
        
        # 특징 벡터 추출
        vector = feature_service.extract_features_from_bytes(file_content)
        
        print(f"✅ 벡터 추출 완료: {vector.shape}")
        
        # 특징 추출 후 retraining 트리거 체크
        trigger_retraining_if_needed()
        
        return JSONResponse({
            "status": "success",
            "feature_vector": vector.tolist(),
            "vector_dimension": len(vector),
            "filename": file.filename,
            "model_info": feature_service.get_vector_info()
        })
        
    except Exception as e:
        print(f"❌ 벡터 추출 실패: {e}")
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)

@app.post("/api/extract_features_from_url/")
async def extract_features_from_url_api(request: ImageUrlRequest):
    """이미지 URL에서 특징 벡터 추출해서 반환 (등록 시스템용 API)"""
    try:
        # 요청에서 이미지 URL 추출
        image_url = request.image_url
        if not image_url:
            return JSONResponse({
                "status": "error",
                "message": "image_url이 필요합니다"
            }, status_code=400)
        
        print(f"🌐 URL에서 벡터 추출 요청: {image_url}")
        
        # URL에서 이미지 다운로드
        import requests
        from urllib.parse import urlparse
        
        # URL 유효성 검사
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return JSONResponse({
                "status": "error",
                "message": "유효하지 않은 URL입니다"
            }, status_code=400)
        
        # 이미지 다운로드
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Content-Type 확인 (이미지인지 검증)
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return JSONResponse({
                "status": "error",
                "message": f"이미지가 아닌 파일입니다. Content-Type: {content_type}"
            }, status_code=400)
        
        print(f"📥 이미지 다운로드 완료: {len(response.content)} bytes")
        
        # 특징 벡터 추출
        vector = feature_service.extract_features_from_bytes(response.content)
        
        print(f"✅ 벡터 추출 완료: {vector.shape}")
        
        # 특징 추출 후 retraining 트리거 체크
        trigger_retraining_if_needed()
        
        return JSONResponse({
            "status": "success",
            "feature_vector": vector.tolist(),
            "vector_dimension": len(vector),
            "image_url": image_url,
            "image_size_bytes": len(response.content),
            "content_type": content_type,
            "model_info": feature_service.get_vector_info()
        })
    except requests.exceptions.RequestException as e:
        print(f"❌ 이미지 다운로드 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"이미지 다운로드 실패: {str(e)}"
        }, status_code=400)
    except Exception as e:
        print(f"❌ 벡터 추출 실패: {e}")
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)

@app.get("/api/feature_service_info/")
async def get_feature_service_info():
    """특징 추출 서비스 정보 반환"""
    try:
        info = feature_service.get_vector_info()
        return JSONResponse({
            "status": "success",
            "service_info": info
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/api/image/{file_path:path}")
async def serve_image(file_path: str):
    """이미지 파일 서빙 (실제 + 더미)"""
    try:
        print("\n[serve_image] --- 요청 ---")
        print(f"file_path(raw): {file_path}")

        # 경로 정규화
        file_path = file_path.replace('/', os.sep)
        print(f"file_path(norm): {file_path}")

        # output_keypoints 경로로 시작하면 직접 파일 생성/서빙 로직 수행 (리다이렉트 금지)
        if file_path.startswith('output_keypoints' + os.sep) or file_path.startswith('output_keypoints/'):
            rel_path = file_path.replace('output_keypoints' + os.sep, '').replace('output_keypoints/', '')
            filename = rel_path.split('/')[-1].split('\\')[-1]
            full_path = os.path.join(OUTPUT_FOLDER, filename)
            print(f"[output_keypoints] 찾는 파일: {filename}")
            print(f"[output_keypoints] 절대경로: {full_path}")
            print(f"[output_keypoints] 실제 파일 존재: {os.path.exists(full_path) and os.path.isfile(full_path)}")
            if os.path.exists(full_path) and os.path.isfile(full_path):
                print(f"📷 (output_keypoints) 이미지 서빙: {full_path}")
                return FileResponse(full_path, media_type="image/jpeg")
            # 없으면 아래 일반 로직으로 진입 (생성 시도)
            file_path = filename

        # 파일명만 추출해서 uploads 폴더에서 먼저 찾기
        filename = file_path.split('/')[-1].split('\\')[-1]
        uploads_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"[uploads] 찾는 파일: {filename}")
        print(f"[uploads] 절대경로: {uploads_path}")
        print(f"[uploads] 실제 파일 존재: {os.path.exists(uploads_path) and os.path.isfile(uploads_path)}")
        if os.path.exists(uploads_path) and os.path.isfile(uploads_path):
            print(f"📷 (uploads) 이미지 서빙: {uploads_path}")
            return FileResponse(uploads_path, media_type="image/jpeg")

        # output_keypoints 폴더에서 찾기 (기존 로직 유지)
        full_path = os.path.join(OUTPUT_FOLDER, filename)
        print(f"[output_keypoints 2차] 절대경로: {full_path}")
        print(f"[output_keypoints 2차] 실제 파일 존재: {os.path.exists(full_path) and os.path.isfile(full_path)}")
        if os.path.exists(full_path) and os.path.isfile(full_path):
            print(f"📷 (output_keypoints 2차) 이미지 서빙: {full_path}")
            return FileResponse(full_path, media_type="image/jpeg")

        # output_keypoints에 파일이 없고, 파일명이 *_keypoints.jpg 형태라면 원본 이미지를 찾아서 생성 시도
        if filename.endswith('_keypoints.jpg'):
            orig_name = filename.replace('_keypoints.jpg', '')
            print(f"[dynamic gen] 원본 추정 이름: {orig_name}")
            from database import get_all_pet_images
            pet_images = get_all_pet_images()
            orig_img_path = None
            for img in pet_images:
                for key in ['public_url', 'image_url', 'image_path', 'file_name']:
                    v = img.get(key)
                    if v and os.path.splitext(os.path.basename(str(v)))[0] == orig_name:
                        orig_img_path = v
                        print(f"[dynamic gen] 원본 이미지 찾음: {orig_img_path} (key: {key})")
                        break
                if orig_img_path:
                    break
            if orig_img_path:
                try:
                    from training.visualize_keypoints import detect_and_visualize_keypoints, setup_ap10k_model
                    global ap10k_model, device, visualizer
                    if ap10k_model is None or device is None or visualizer is None:
                        print("[dynamic gen] AP10K 모델 재로딩 시도")
                        ap10k_model, device, visualizer = setup_ap10k_model()
                    print(f"[dynamic gen] detect_and_visualize_keypoints 호출: {orig_img_path}")
                    output_path, _ = detect_and_visualize_keypoints(orig_img_path, ap10k_model, device, visualizer)
                    print(f"[dynamic gen] output_path: {output_path}")
                    if output_path and os.path.exists(output_path):
                        print(f"📷 (dynamic gen) 생성된 이미지 서빙: {output_path}")
                        return FileResponse(output_path, media_type="image/jpeg")
                    else:
                        print(f"[dynamic gen] output_path 파일 없음: {output_path}")
                except Exception as e:
                    print(f"[dynamic gen] 예외 발생: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[dynamic gen] 원본 이미지 DB에서 찾지 못함")

        # 파일을 찾지 못했을 경우 더미 이미지 생성 (uploads 폴더에 생성)
        print(f"[dummy] 더미 이미지 생성 시도 (keypoint in file_path: {'keypoint' in file_path.lower()})")
        if 'keypoint' in file_path.lower():
            dummy_img = Image.new('RGB', (400, 400), color=(50, 50, 50))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(dummy_img)
            keypoints = [(100, 100), (150, 80), (200, 120), (180, 200), (120, 220)]
            for kp in keypoints:
                draw.ellipse([kp[0]-5, kp[1]-5, kp[0]+5, kp[1]+5], fill='red')
            connections = [(0,1), (1,2), (2,3), (3,4)]
            for conn in connections:
                draw.line([keypoints[conn[0]], keypoints[conn[1]]], fill='yellow', width=2)
        else:
            dummy_img = Image.new('RGB', (224, 224), color=(150, 100, 50))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(dummy_img)
            draw.text((50, 100), "강아지 이미지", fill='white')

        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}.jpg")
        dummy_img.save(temp_path, 'JPEG')
        print(f"[dummy] 더미 이미지 저장: {temp_path}")
        return FileResponse(temp_path, media_type="image/jpeg")

    except Exception as e:
        print(f"❌ 이미지 서빙 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """시스템 상태 체크"""
    status = {
        "status": "healthy",
        "models_available": MODELS_AVAILABLE,
        "ap10k_model_loaded": ap10k_model is not None,
        "mode": "real_model" if (MODELS_AVAILABLE and ap10k_model is not None) else "dummy"
    }
    
    if MODELS_AVAILABLE:
        status["simclr_model_path"] = SIMCLR_MODEL_PATH
        status["message"] = "실제 모델 사용 가능" if ap10k_model is not None else "모델 로드 대기 중"
    else:
        status["message"] = "더미 모드 - 모델 모듈 없음"
    
    return JSONResponse(status)

def convert_breed_codes_in_dog_data(dog_data):
    """강아지 데이터에서 견종 코드를 견종명으로 변환"""
    if isinstance(dog_data, list):
        return [convert_breed_codes_in_dog_data(dog) for dog in dog_data]
    elif isinstance(dog_data, dict):
        dog_copy = dog_data.copy()
        
        # 견종 코드 변환 (백엔드에서 처리)
        if 'breed' in dog_copy and dog_copy['breed']:
            breed_name = get_breed_name_by_code(dog_copy['breed'])
            dog_copy['breed'] = breed_name
            dog_copy['breed_code'] = dog_data['breed']  # 원본 코드도 보존
        
        # 성별과 입양상태는 원본 코드 그대로 전달 (프론트엔드에서 변환)
        if 'gender' in dog_copy:
            dog_copy['gender_code'] = dog_copy['gender']  # 원본 코드 보존
        
        if 'adoption_status' in dog_copy:
            dog_copy['adoption_status_code'] = dog_copy['adoption_status']  # 원본 코드 보존
        
        return dog_copy
    return dog_data

@app.get("/api/dogs/")
async def get_all_dogs_api():
    """실제 DB에서 모든 강아지 정보 조회"""
    try:
        dogs = get_all_dogs()
        # 견종 코드를 견종명으로 변환
        dogs_with_breed_names = convert_breed_codes_in_dog_data(dogs)
        
        return JSONResponse({
            "status": "success",
            "dogs": dogs_with_breed_names,
            "total_count": len(dogs_with_breed_names)
        })
        
    except Exception as e:
        print(f"❌ DB 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/api/dogs/{dog_id}")
async def get_dog_detail_api(dog_id: int):
    """특정 강아지 상세 정보 조회"""
    try:
        dog = get_dog_by_id(dog_id)
        if not dog:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
        
        # 견종 코드를 견종명으로 변환
        dog_with_breed_name = convert_breed_codes_in_dog_data(dog)
        
        return JSONResponse({
            "status": "success", 
            "dog": dog_with_breed_name
        })
        
    except Exception as e:
        print(f"❌ 강아지 상세 정보 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Pydantic 모델 정의
class ImageUrlRequest(BaseModel):
    image_url: str

# Pydantic 모델 추가
class DogCreateRequest(BaseModel):
    name: str
    breed: str
    age: int
    gender: str
    size: str
    weight: str = None
    location: str
    description: str = None
    image_url: str = None
    additional_images: list = []
    health_info: str = None
    vaccination: str = None
    neutered: bool = False
    contact: str = None
    contact_name: str = None
    shelter_name: str = None
    adoption_status: str = "입양 가능"

@app.post("/api/dogs/")
async def add_dog_api(dog_data: DogCreateRequest):
    """새 강아지 정보 추가"""
    try:
        dog_id = add_dog(dog_data.dict())
        return JSONResponse({
            "status": "success",
            "message": "강아지 정보가 추가되었습니다",
            "dog_id": dog_id
        })
    except Exception as e:
        print(f"❌ 강아지 추가 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.put("/api/dogs/{dog_id}")
async def update_dog_api(dog_id: int, dog_data: DogCreateRequest):
    """강아지 정보 수정"""
    try:
        success = update_dog(dog_id, dog_data.dict())
        if success:
            return JSONResponse({
                "status": "success",
                "message": "강아지 정보가 수정되었습니다"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
    except Exception as e:
        print(f"❌ 강아지 수정 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.delete("/api/dogs/{dog_id}")
async def delete_dog_api(dog_id: int):
    """강아지 정보 삭제"""
    try:
        success = delete_dog(dog_id)
        if success:
            return JSONResponse({
                "status": "success",
                "message": "강아지 정보가 삭제되었습니다"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
    except Exception as e:
        print(f"❌ 강아지 삭제 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/dogs/{dog_id}/image_mapping")
async def add_image_mapping_api(dog_id: int, image_path: str, feature_vector: list = None):
    """강아지-이미지 매핑 추가"""
    try:
        mapping_id = add_image_mapping(dog_id, image_path, feature_vector)
        return JSONResponse({
            "status": "success",
            "message": "이미지 매핑이 추가되었습니다",
            "mapping_id": mapping_id
        })
    except Exception as e:
        print(f"❌ 이미지 매핑 추가 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/search_with_db_mapping/")
async def search_with_db_mapping(file: UploadFile = File(...)):
    """유사도 검색 + DB 정보 매핑"""
    try:
        # 1. 기존 유사도 검색 수행
        search_response = await upload_and_search(file)
        search_data = search_response.body.decode('utf-8')
        import json
        search_result = json.loads(search_data)
        
        if not search_result.get('success'):
            return search_response
        
        # 2. 검색 결과를 실제 DB 정보로 매핑
        mapped_results = []
        for i, result in enumerate(search_result.get('results', [])):
            # 이미지 경로로 실제 DB에서 강아지 정보 조회
            image_path = result.get('image_path', '')
            db_dog = get_dog_by_image_path(image_path)
            
            if db_dog:
                # DB에서 찾은 경우 실제 정보 사용
                db_dog_info = {
                    "id": db_dog['id'],
                    "name": db_dog['name'],
                    "breed": db_dog['breed'],
                    "age": db_dog['age'],
                    "gender": db_dog['gender'],
                    "size": db_dog['size'],
                    "location": db_dog['location'],
                    "description": db_dog['description'],
                    "image_url": db_dog['image_url'],
                    "contact": db_dog['contact'],
                    "adoption_status": db_dog['adoption_status']
                }
            else:
                # DB에서 찾지 못한 경우 임시 정보 생성 (나중에 매핑 필요)
                dog_id = i + 1
                db_dog_info = {
                    "id": dog_id,
                    "name": f"강아지 {dog_id}",
                    "breed": ["골든 리트리버", "믹스견", "래브라도", "비글", "포메라니안"][i % 5],
                    "age": (i % 5) + 1,
                    "gender": "수컷" if i % 2 == 0 else "암컷",
                    "size": ["대형견", "중형견", "소형견"][i % 3],
                    "location": ["서울 강남구", "서울 송파구", "경기 성남시", "인천 부평구", "서울 마포구"][i % 5],
                    "description": f"유사도 {result.get('combined_similarity', 0):.2f}의 강아지입니다. (DB 매핑 필요)",
                    "image_url": f"https://example.com/dog{dog_id}.jpg",
                    "contact": f"010-{1000+i*111:04d}-{5678+i*111:04d}",
                    "adoption_status": "입양 가능"
                }
            
            # 유사도 정보와 DB 정보 결합
            mapped_result = {
                **result,  # 기존 유사도 정보
                **db_dog_info,  # DB 강아지 정보
                "similarity_score": result.get('combined_similarity', 0)
            }
            
            mapped_results.append(mapped_result)
        
        return JSONResponse({
            'success': True,
            'query_image': search_result.get('query_image'),
            'query_keypoint_image': search_result.get('query_keypoint_image'),
            'results': mapped_results,
            'mode': search_result.get('mode'),
            'total_found': len(mapped_results)
        })
        
    except Exception as e:
        print(f"❌ DB 매핑 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/db-tables")
async def get_db_tables():
    """데이터베이스 테이블 목록 조회 (디버그용)"""
    try:
        db = DatabaseManager()
        tables = db.show_tables()
        return {"success": True, "tables": tables}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/debug/db-table-structure/{table_name}")
async def get_table_structure(table_name: str):
    """특정 테이블의 구조 조회 (디버그용)"""
    try:
        db = DatabaseManager()
        structure = db.describe_table(table_name)
        return {"success": True, "table": table_name, "structure": structure}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/breed_codes/")
async def get_breed_codes_api():
    """견종 코드 목록 조회"""
    try:
        breed_codes = get_breed_codes()
        return JSONResponse({
            "status": "success",
            "breed_codes": breed_codes,
            "total_count": len(breed_codes)
        })
    except Exception as e:
        print(f"❌ 견종 코드 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/")
async def root():
    return {"message": "Dog Similarity Search API"}

# feature_extraction_service의 FastAPI router 등록
from feature_extraction_service import router as feature_router
app.include_router(feature_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
