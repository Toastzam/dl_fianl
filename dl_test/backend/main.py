from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
import json
import tempfile
import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTImageProcessor
import cv2
from matplotlib import cm

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 우리가 만든 모듈 임포트
from training.visualize_keypoints import (
    setup_ap10k_model, 
    detect_and_visualize_keypoints,
    calculate_keypoint_similarity
)
from training.search_similar_dogs import search_similar_dogs

class SimCLREncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", projection_dim=128):
        super().__init__()
        config = ViTConfig.from_pretrained(model_name)
        # ViTModel을 사용하면, 내부적으로 어텐션 가중치를 추출하기 위한 hook을 걸 수 있습니다.
        # 실제 어텐션 맵 시각화를 위해서는 더 복잡한 로직이 필요합니다.
        self.vit = ViTModel.from_pretrained(model_name, output_attentions=True) # 어텐션 가중치 출력 활성화
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, projection_dim)
        )
        self.model_name = model_name

    def forward(self, pixel_values):
        # ViT의 마지막 hidden state의 CLS 토큰만 사용
        # output_attentions=True 설정하면 outputs.attentions 에 어텐션 가중치도 반환됨
        outputs = self.vit(pixel_values=pixel_values)
        vit_output = outputs.last_hidden_state[:, 0, :] # CLS token output
        projection = self.projection_head(vit_output)
        return projection, outputs.attentions # 어텐션 가중치도 함께 반환

# --- FastAPI 앱 초기화 ---

# --- 전역 변수 설정 ---
model = None
processor = None
ap10k_model = None
ap10k_device = None
visualizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 업로드 폴더 설정
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("output_keypoints", exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 서버 시작 시 모델 로드 (앱이 처음 실행될 때 한 번만) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 로드
    global ap10k_model, device, visualizer
    try:
        print("🚀 AP-10K 모델 로딩 중...")
        ap10k_model, device, visualizer = setup_ap10k_model()
        print("✅ AP-10K 모델 로드 완료!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        ap10k_model = None
    
    yield
    
    # 종료 시 정리
    print("🔄 서버 종료 중...")

# FastAPI 앱 생성 시 lifespan 사용
app = FastAPI(
    title="Dog Similarity Search API",
    description="SimCLR + AP-10K 키포인트 기반 강아지 유사도 검색 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (이미지) 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output_keypoints", StaticFiles(directory="output_keypoints"), name="output_keypoints")

# --- 히트맵 생성 함수 (매우 간략화된 예시) ---
# 실제 어텐션 맵 시각화를 위해서는 ViTModel의 outputs.attentions를 사용하여
# 각 헤드의 어텐션 가중치를 조합하고, 이를 이미지 크기에 맞게 업샘플링해야 합니다.
# 이는 복잡한 작업이므로, 현재는 시각화 효과를 위한 "더미" 히트맵을 생성합니다.
def generate_dummy_heatmap_b64(image_pil: Image.Image, value=0.5):
    # 입력 이미지와 동일한 크기의 히트맵 데이터를 생성 (0.0 ~ 1.0)
    heatmap_data = np.full((image_pil.height, image_pil.width), value, dtype=np.float32)

    # 중앙에 밝은 원을 그리는 예시 (실제 어텐션 맵이 아님)
    center_x, center_y = image_pil.width // 2, image_pil.height // 2
    radius = min(image_pil.width, image_pil.height) // 4
    cv2.circle(heatmap_data, (center_x, center_y), radius, 1.0, -1) # 원형으로 강조

    # 히트맵 데이터를 컬러맵으로 변환 (matplotlib의 jet 컬러맵 사용)
    # RGBA (Red, Green, Blue, Alpha) 형태로 나옴
    heatmap_colored = cm.jet(heatmap_data)
    # 0-1 스케일의 RGBA를 0-255 스케일의 RGBA로 변환
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)

    # PIL Image 객체로 변환
    heatmap_pil = Image.fromarray(heatmap_colored_uint8, 'RGBA')

    # PIL Image를 Bytes로 변환 (PNG 형식)
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    heatmap_bytes = buffer.getvalue()
    return base64.b64encode(heatmap_bytes).decode('utf-8')

# 기존 generate_dummy_heatmap_b64는 임시로 남겨두고,
# 실제 어텐션 맵을 위한 함수를 새로 만듭니다.
# 주의: 이 코드는 개념 설명이며, 실제 동작하는 완전한 코드는 아닙니다.
# ViT의 내부 구조와 attention map 추출 방법에 대한 깊은 이해가 필요합니다.
def get_attention_heatmap(original_image_pil: Image.Image, attentions, patch_size=16):
    # attentions는 outputs.attentions 입니다.
    # 일반적으로 ViT의 attentions는 (num_layers, batch_size, num_heads, seq_len, seq_len) 형태입니다.
    # 우리는 마지막 레이어의 어텐션 가중치를 사용하고,
    # 특히 [CLS] 토큰이 다른 패치들에 얼마나 집중했는지를 시각화하는 것이 일반적입니다.

    # 1. CLS 토큰에 대한 어텐션 가중치 추출 (예시)
    # outputs.attentions는 튜플 형태이므로, 마지막 레이어 ([0] 인덱스)를 가져옵니다.
    # attentions_last_layer = attentions[-1] # (batch_size, num_heads, seq_len, seq_len)

    # 모든 헤드의 CLS 토큰 (인덱스 0)이 다른 패치들 (인덱스 1부터)에 대한 평균 어텐션 가중치를 계산
    # 예: attention_weights = attentions_last_layer[0, :, 0, 1:].mean(dim=0)
    # (이 부분은 모델의 정확한 구조와 원하는 시각화 방식에 따라 달라집니다.)
    
    # -------------------------------------------------------------
    # 여기서는 임시로 ViT 모델 내부의 Attention 가중치를 직접 접근하는 예시를 보여드립니다.
    # 실제로는 model.vit.encoder.layer[-1].attention.self.get_attn_map() 같은
    # 특정 hook이나 메서드를 통해 가져오는 것이 더 안정적일 수 있습니다.
    # DINO 모델 같은 경우는 공식 깃허브에 시각화 코드가 잘 제공됩니다.
    # -------------------------------------------------------------

    # **가장 간단하게 ViT에서 어텐션 맵을 얻는 방법 중 하나 (개념적 코드):**
    # ViTModel.forward 시 output_attentions=True 설정하면 outputs.attentions가 반환됩니다.
    # outputs.attentions는 각 레이어의 어텐션 텐서들의 튜플입니다.
    # 가장 마지막 레이어의 어텐션 가중치 (outputs.attentions[-1])를 사용합니다.
    # 이 가중치는 (batch_size, num_heads, sequence_length, sequence_length) 형태입니다.
    # 여기서 sequence_length는 1 (CLS 토큰) + (이미지 패치 수) 입니다.
    
    # CLS 토큰 (인덱스 0)이 다른 패치들에 대한 어텐션 가중치만 가져옵니다.
    # 모든 헤드의 평균을 사용하는 것이 일반적입니다.
    # 예시: attention_map = attentions[-1][0, :, 0, 1:].mean(dim=0) # [num_patches]
    
    # 지금은 실제 어텐션 가중치 추출 로직 대신, '더미'로 대체합니다.
    # 이 부분을 직접 구현하셔야 합니다!
    
    # **임시로, 다시 더미 히트맵을 생성하여 전달합니다.**
    # (사용자님이 이 함수를 실제 어텐션 맵 생성 로직로 교체해야 함)
    # -----------------------------------------------------------------
    
    # 실제 어텐션 맵 데이터 (0~1 사이)
    # 이 부분에 위에서 계산한 `attention_map`을 사용해야 합니다.
    # attention_map = attention_map.cpu().numpy().reshape(image_width // patch_size, image_height // patch_size)
    # 이후 resize하여 원본 이미지 크기로 맞춤

    # 현재는 아래 generate_dummy_heatmap_b64를 계속 사용합니다.
    # 따라서 계속 중앙에 원형 히트맵이 나올 것입니다.
    return generate_dummy_heatmap_b64(original_image_pil) # <<< 이 부분을 실제 로직으로 교체해야 함


def get_attention_heatmap(original_image_pil: Image.Image, attentions, patch_size=16):
    heatmap_data = np.full((original_image_pil.height, original_image_pil.width), 0.5, dtype=np.float32)
    center_x, center_y = original_image_pil.width // 2, original_image_pil.height // 2
    radius = min(original_image_pil.width, original_image_pil.height) // 4
    cv2.circle(heatmap_data, (center_x, center_y), radius, 1.0, -1)
    heatmap_colored = cm.jet(heatmap_data)
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored_uint8, 'RGBA')
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    heatmap_bytes = buffer.getvalue()
    heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
    return heatmap_b64, heatmap_data


# --- API 엔드포인트 정의 ---
@app.post("/compare_dogs_with_heatmap/")
async def compare_dog_images_with_heatmap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"message": "모델이 아직 로드되지 않았거나 오류가 발생했습니다."})

    try:
        # 1. 이미지 읽기 (PIL Image 객체로)
        image1_pil = Image.open(io.BytesIO(await file1.read())).convert("RGB")
        image2_pil = Image.open(io.BytesIO(await file2.read())).convert("RGB")

        # 2. 이미지 전처리 및 모델 입력 준비
        # ViTImageProcessor를 사용하면 편리하게 전처리 가능
        inputs1 = processor(images=image1_pil, return_tensors="pt").to(device)
        inputs2 = processor(images=image2_pil, return_tensors="pt").to(device)

        # 3. 모델 순전파 (특징 추출 및 어텐션 가중치 얻기)
        with torch.no_grad():
            features1, attentions1 = model(inputs1['pixel_values'])
            features2, attentions2 = model(inputs2['pixel_values'])

        # 4. 유사도 계산
        # SimCLR의 projection_head 출력을 사용해야 하지만,
        # 유사도 시각화를 위해 feature(CLS token output)를 사용할 수도 있습니다.
        # 여기서는 features (CLS token output) 사용
        similarity = F.cosine_similarity(features1, features2).item()

        # 5. 히트맵 생성 및 Base64 인코딩
        # 실제 어텐션 맵을 사용하는 함수로 교체 필요!
        heatmap_b64_1, heatmap1 = get_attention_heatmap(image1_pil, attentions1)
        heatmap_b64_2, heatmap2 = get_attention_heatmap(image2_pil, attentions2)
        
        # 예시: 더미 좌표 (중앙)
        point1 = {"x": image1_pil.width // 2, "y": image1_pil.height // 2}
        point2 = {"x": image2_pil.width // 2, "y": image2_pil.height // 2}

        y1, x1 = np.unravel_index(np.argmax(heatmap1), heatmap1.shape)
        y2, x2 = np.unravel_index(np.argmax(heatmap2), heatmap2.shape)
        point1 = {"x": int(x1), "y": int(y1)}
        point2 = {"x": int(x2), "y": int(y2)}

        return JSONResponse({
            "similarity": similarity,
            "heatmap_image1": f"data:image/png;base64,{heatmap_b64_1}",
            "heatmap_image2": f"data:image/png;base64,{heatmap_b64_2}",
            "point1": point1,  # 첫 번째 이미지의 유사 부위 좌표
            "point2": point2,  # 두 번째 이미지의 유사 부위 좌표
            "message": "강아지 유사도 비교 및 히트맵 생성 완료!"
        })

    except Exception as e:
        print(f"API 처리 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"message": f"서버 오류: {str(e)}"})

# --- 새로운 API: 이미지 업로드 및 유사도 검색 ---
@app.post("/api/upload_and_search/")
async def upload_and_search_similar_dogs(file: UploadFile = File(...)):
    """이미지 업로드 및 유사도 검색 API"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="SimCLR 모델이 로드되지 않았습니다.")
    
    if ap10k_model is None:
        raise HTTPException(status_code=503, detail="AP-10K 키포인트 모델이 로드되지 않았습니다.")
    
    try:
        # 1. 업로드된 이미지 저장
        file_content = await file.read()
        filename = f"query_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as f:
            f.write(file_content)
        
        print(f"🔍 쿼리 이미지 저장: {filepath}")
        
        # 2. 쿼리 이미지 키포인트 검출
        query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
            filepath, ap10k_model, ap10k_device, visualizer
        )
        
        if query_pose_results is None:
            raise HTTPException(status_code=500, detail="키포인트 검출에 실패했습니다.")
        
        # 3. SimCLR 기반 유사 이미지 검색
        print("🔍 SimCLR 기반 유사 이미지 검색...")
        similar_results = search_similar_dogs(
            query_image_path=filepath,
            top_k=5,
            model_path="../models/simclr_vit_dog_model.pth",
            out_dim=128,
            image_size=224,
            db_features_file="../db_features.npy",
            db_image_paths_file="../db_image_paths.npy"
        )
        
        # 4. 각 유사 이미지에 대해 키포인트 검출 및 유사도 계산
        results = []
        for i, (simclr_similarity, similar_path) in enumerate(similar_results):
            print(f"🔍 유사 이미지 {i+1} 키포인트 검출: {similar_path}")
            
            # 키포인트 검출
            similar_kp_output_path, similar_pose_results = detect_and_visualize_keypoints(
                similar_path, ap10k_model, ap10k_device, visualizer
            )
            
            # 키포인트 유사도 계산
            keypoint_similarity = 0.0
            if similar_pose_results is not None:
                keypoint_similarity = calculate_keypoint_similarity(
                    query_pose_results, similar_pose_results
                )
            
            # 복합 유사도 계산 (SimCLR 70% + 키포인트 30%)
            combined_similarity = (0.7 * simclr_similarity) + (0.3 * keypoint_similarity)
            
            results.append({
                'rank': i + 1,
                'image_path': similar_path.replace('\\', '/'),
                'keypoint_image_path': similar_kp_output_path.replace('\\', '/') if similar_kp_output_path else None,
                'simclr_similarity': float(simclr_similarity),
                'keypoint_similarity': float(keypoint_similarity),
                'combined_similarity': float(combined_similarity)
            })
            
            print(f"  ✅ SimCLR: {simclr_similarity:.4f}, 키포인트: {keypoint_similarity:.4f}, 복합: {combined_similarity:.4f}")
        
        # 복합 유사도로 재정렬
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        return JSONResponse({
            'success': True,
            'query_image': filepath.replace('\\', '/'),
            'query_keypoint_image': query_kp_output_path.replace('\\', '/') if query_kp_output_path else None,
            'results': results
        })
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.get("/api/image/{file_path:path}")
async def serve_image(file_path: str):
    """이미지 파일 서빙"""
    try:
        # 경로 정규화
        file_path = file_path.replace('/', os.sep)
        
        # 다양한 경로에서 이미지 찾기
        possible_paths = [
            file_path,
            os.path.join('uploads', file_path),
            os.path.join('output_keypoints', file_path),
            os.path.join('training', file_path),
            os.path.join('..', file_path)  # 상위 디렉토리도 검색
        ]
        
        for path in possible_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                print(f"📷 이미지 서빙: {full_path}")
                return FileResponse(full_path)
        
        print(f"❌ 이미지를 찾을 수 없음: {file_path}")
        print(f"시도한 경로들: {possible_paths}")
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
        
    except Exception as e:
        print(f"❌ 이미지 서빙 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 헬스 체크 API ---
@app.get("/health")
async def health_check():
    """헬스 체크"""
    return JSONResponse({
        'status': 'healthy', 
        'simclr_model_loaded': model is not None,
        'ap10k_model_loaded': ap10k_model is not None
    })