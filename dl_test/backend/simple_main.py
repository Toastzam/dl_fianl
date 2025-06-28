from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
import tempfile
import shutil
from PIL import Image
import random

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 실제 모델 임포트 (에러 처리 포함)
ap10k_model = None
device = None
visualizer = None
search_similar_dogs = None

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
app = FastAPI(
    title="Dog Similarity Search API", 
    description="SimCLR + AP-10K 키포인트 기반 강아지 유사도 검색",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files 마운트 - 이미지 서빙을 위해 추가
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output_keypoints", StaticFiles(directory="output_keypoints"), name="output_keypoints")
app.mount("/training", StaticFiles(directory="../training"), name="training")

# 업로드 폴더 설정
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_keypoints"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# SimCLR 관련 설정
SIMCLR_MODEL_PATH = '../models/simclr_vit_dog_model.pth'
SIMCLR_OUT_DIM = 128  # 실제 저장된 모델과 일치하도록 복원
SIMCLR_IMAGE_SIZE = 224
DB_FEATURES_FILE = '../db_features.npy'
DB_IMAGE_PATHS_FILE = '../db_image_paths.npy'

@app.post("/api/upload_and_search/")
async def upload_and_search(file: UploadFile = File(...)):
    """실제 업로드 및 유사도 검색 API"""
    try:
        # 파일 저장
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
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
            return await real_model_search(file_location, file.filename)
        else:
            # 더미 모드
            print("🔄 더미 모드로 진입")
            return await dummy_search(file_location, file.filename)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def real_model_search(file_location: str, filename: str):
    """실제 모델을 사용한 검색"""
    try:
        print("🔍 실제 모델 사용 - 시작")
        
        # 1. 쿼리 이미지 키포인트 검출
        print("🔍 단계 1: 쿼리 이미지 키포인트 검출")
        query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
            file_location, ap10k_model, device, visualizer
        )
        print(f"✅ 쿼리 이미지 키포인트 검출 완료: {query_kp_output_path}")
        
        # 2. SimCLR 기반 유사 이미지 검색 (여기서 업로드된 이미지의 특징 추출도 수행됨)
        print("🔍 단계 2: SimCLR 기반 유사 이미지 검색 시작...")
        similar_results = search_similar_dogs(
            query_image_path=file_location,
            top_k=5,
            model_path=SIMCLR_MODEL_PATH,
            out_dim=SIMCLR_OUT_DIM,
            image_size=SIMCLR_IMAGE_SIZE,
            db_features_file=DB_FEATURES_FILE,
            db_image_paths_file=DB_IMAGE_PATHS_FILE
        )
        print(f"✅ SimCLR 검색 완료: {len(similar_results)}개 결과")
        
        # 3. 각 유사 이미지에 대해 키포인트 검출 및 유사도 계산
        print("🔍 단계 3: 유사 이미지들의 키포인트 검출 및 종합 유사도 계산")
        results = []
        for i, (simclr_score, similar_path) in enumerate(similar_results):
            print(f"  🔍 유사 이미지 {i+1}/{len(similar_results)} 처리: {os.path.basename(similar_path)}")
            
            # 경로 정규화 (Windows 경로 문제 해결)
            similar_path_normalized = similar_path.replace('\\', '/')
            full_similar_path = os.path.join('..', similar_path_normalized.replace('/', os.sep))
            
            if not os.path.exists(full_similar_path):
                print(f"⚠️ 이미지 파일을 찾을 수 없음: {full_similar_path}")
                # 파일이 없으면 키포인트 유사도 0으로 설정
                keypoint_similarity = 0.0
                similar_kp_output_path = None
            else:
                # 키포인트 검출
                try:
                    similar_kp_output_path, similar_pose_results = detect_and_visualize_keypoints(
                        full_similar_path, ap10k_model, device, visualizer
                    )
                    
                    # 키포인트 유사도 계산
                    keypoint_similarity = 0.0
                    if query_pose_results and similar_pose_results:
                        keypoint_similarity = calculate_keypoint_similarity(
                            query_pose_results, similar_pose_results
                        )
                except Exception as e:
                    print(f"⚠️ 키포인트 검출 실패 ({os.path.basename(similar_path)}): {e}")
                    keypoint_similarity = 0.0
                    similar_kp_output_path = None
            
            # 복합 유사도 계산 (SimCLR 70% + 키포인트 30%)
            combined_similarity = (0.7 * simclr_score) + (0.3 * keypoint_similarity)
            
            results.append({
                'rank': i + 1,
                'image_path': similar_path_normalized,
                'keypoint_image_path': similar_kp_output_path.replace('\\', '/') if similar_kp_output_path else None,
                'simclr_similarity': float(simclr_score),
                'keypoint_similarity': float(keypoint_similarity),
                'combined_similarity': float(combined_similarity)
            })
            
            print(f"    ✅ SimCLR: {simclr_score:.4f}, 키포인트: {keypoint_similarity:.4f}, 복합: {combined_similarity:.4f}")
        
        # 복합 유사도로 재정렬
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        print("🔍 단계 4: 복합 유사도로 재정렬 완료")
        
        # 순위 업데이트
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return JSONResponse({
            'success': True,
            'query_image': file_location.replace('\\', '/'),
            'query_keypoint_image': query_kp_output_path.replace('\\', '/') if query_kp_output_path else None,
            'results': results,
            'mode': 'real_model'
        })
        
    except Exception as e:
        print(f"❌ 실제 모델 검색 중 오류: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 더미 모드로 폴백")
        return await dummy_search(file_location, filename)

async def dummy_search(file_location: str, filename: str):
    """더미 데이터를 사용한 검색 (테스트/폴백용)"""
    print("🔄 더미 모드 사용")
    
    # 실제 존재하는 이미지 파일들로 더미 검색 결과 생성
    real_images = [
        'training/Images/n02087394-Rhodesian_ridgeback/n02087394_11149.jpg',
        'training/Images/n02087394-Rhodesian_ridgeback/n02087394_1137.jpg',
        'training/Images/n02087394-Rhodesian_ridgeback/n02087394_1336.jpg',
        'training/Images/n02087394-Rhodesian_ridgeback/n02087394_1352.jpg',
        'training/Images/n02087394-Rhodesian_ridgeback/n02087394_1706.jpg',
        'training/Images/n02088238-basset/n02088238_1261.jpg',
        'training/Images/n02091134-whippet/n02091134_9793.jpg',
        'training/Images/n02090379-redbone/n02090379_2420.jpg'
    ]
    
    dummy_results = []
    for i in range(min(8, len(real_images))):
        image_path = real_images[i]
        # 키포인트 이미지 경로 생성
        image_name = image_path.split('/')[-1].replace('.jpg', '_keypoints.jpg')
        
        dummy_results.append({
            'rank': i + 1,
            'image_path': image_path,
            'keypoint_image_path': f'output_keypoints/{image_name}',
            'simclr_similarity': random.uniform(0.7, 0.95),
            'keypoint_similarity': random.uniform(0.6, 0.9),
            'combined_similarity': random.uniform(0.65, 0.92)
        })
    
    # 복합 유사도로 정렬
    dummy_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
    
    return JSONResponse({
        'success': True,
        'query_image': file_location.replace('\\', '/'),
        'query_keypoint_image': f'output_keypoints/{filename}_keypoints.jpg',
        'results': dummy_results,
        'mode': 'dummy'
    })

@app.get("/api/image/{file_path:path}")
async def serve_image(file_path: str):
    """이미지 파일 서빙 (실제 + 더미)"""
    try:
        # 경로 정규화
        file_path = file_path.replace('/', os.sep)
        
        # 다양한 경로에서 이미지 찾기
        possible_paths = [
            file_path,
            os.path.join('uploads', file_path),
            os.path.join('output_keypoints', file_path),
            os.path.join('../training', file_path),
            os.path.join('../uploads', file_path),
            os.path.join('../output_keypoints', file_path),
            os.path.join('..', file_path)
        ]
        
        # 실제 파일 찾기
        for path in possible_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                print(f"📷 이미지 서빙: {full_path}")
                return FileResponse(full_path)
        
        # 파일을 찾지 못했을 경우 더미 이미지 생성
        print(f"⚠️ 이미지를 찾을 수 없음: {file_path}")
        print(f"시도한 경로들: {possible_paths}")
        
        # 더미 이미지 생성 (키포인트 이미지 스타일)
        if 'keypoint' in file_path.lower():
            # 키포인트 스타일 더미 이미지
            dummy_img = Image.new('RGB', (400, 400), color=(50, 50, 50))
            # 간단한 키포인트 시뮬레이션 (원과 선)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(dummy_img)
            
            # 더미 키포인트 (원들)
            keypoints = [(100, 100), (150, 80), (200, 120), (180, 200), (120, 220)]
            for kp in keypoints:
                draw.ellipse([kp[0]-5, kp[1]-5, kp[0]+5, kp[1]+5], fill='red')
            
            # 더미 골격선
            connections = [(0,1), (1,2), (2,3), (3,4)]
            for conn in connections:
                draw.line([keypoints[conn[0]], keypoints[conn[1]]], fill='yellow', width=2)
                
        else:
            # 일반 강아지 더미 이미지
            dummy_img = Image.new('RGB', (224, 224), color=(150, 100, 50))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(dummy_img)
            draw.text((50, 100), "강아지 이미지", fill='white')
        
        temp_path = f"temp_{os.path.basename(file_path)}.jpg"
        dummy_img.save(temp_path, 'JPEG')
        return FileResponse(temp_path)
        
    except Exception as e:
        print(f"❌ 이미지 서빙 오류: {e}")
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
        status["db_features_file"] = DB_FEATURES_FILE
        status["message"] = "실제 모델 사용 가능" if ap10k_model is not None else "모델 로드 대기 중"
    else:
        status["message"] = "더미 모드 - 모델 모듈 없음"
    
    return JSONResponse(status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
