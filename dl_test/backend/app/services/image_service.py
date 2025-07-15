"""
이미지 관련 비즈니스 로직
"""
import os
import tempfile
from PIL import Image, ImageDraw
from fastapi import HTTPException
from fastapi.responses import FileResponse
from typing import Optional

from app.config import settings
from app.dependencies import get_models

class ImageService:
    def __init__(self):
        pass
    
    def serve_image(self, file_path: str) -> FileResponse:
        """이미지 파일 서빙"""
        try:
            print(f"\n[serve_image] 요청 경로: {file_path}")
            
            # 경로 정규화
            file_path = file_path.replace('/', os.sep)
            
            # output_keypoints 경로 처리
            if file_path.startswith('output_keypoints' + os.sep) or file_path.startswith('output_keypoints/'):
                return self._serve_output_keypoints_image(file_path)
            
            # 파일명 추출
            filename = file_path.split('/')[-1].split('\\')[-1]
            
            # uploads 폴더에서 찾기
            uploads_path = os.path.join(settings.UPLOAD_FOLDER, filename)
            if os.path.exists(uploads_path) and os.path.isfile(uploads_path):
                print(f"📷 (uploads) 이미지 서빙: {uploads_path}")
                return FileResponse(uploads_path, media_type="image/jpeg")
            
            # output_keypoints 폴더에서 찾기
            full_path = os.path.join(settings.OUTPUT_FOLDER, filename)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                print(f"📷 (output_keypoints) 이미지 서빙: {full_path}")
                return FileResponse(full_path, media_type="image/jpeg")
            
            # 동적 키포인트 이미지 생성 시도
            if filename.endswith('_keypoints.jpg'):
                dynamic_path = self._try_generate_keypoint_image(filename)
                if dynamic_path:
                    return FileResponse(dynamic_path, media_type="image/jpeg")
            
            # 더미 이미지 생성
            return self._create_dummy_image(filename)
        
        except Exception as e:
            print(f"❌ 이미지 서빙 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _serve_output_keypoints_image(self, file_path: str) -> Optional[FileResponse]:
        """output_keypoints 이미지 서빙"""
        rel_path = file_path.replace('output_keypoints' + os.sep, '').replace('output_keypoints/', '')
        filename = rel_path.split('/')[-1].split('\\')[-1]
        full_path = os.path.join(settings.OUTPUT_FOLDER, filename)
        
        print(f"[output_keypoints] 파일: {filename}, 경로: {full_path}")
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            print(f"📷 (output_keypoints 직접) 이미지 서빙: {full_path}")
            return FileResponse(full_path, media_type="image/jpeg")
        
        return None
    
    def _try_generate_keypoint_image(self, filename: str) -> Optional[str]:
        """동적 키포인트 이미지 생성"""
        if not filename.endswith('_keypoints.jpg'):
            return None
        
        orig_name = filename.replace('_keypoints.jpg', '')
        print(f"[dynamic gen] 원본 추정 이름: {orig_name}")
        
        try:
            from app.database import get_all_pet_images
            pet_images = get_all_pet_images()
            
            orig_img_path = None
            for img in pet_images:
                for key in ['public_url', 'image_url', 'image_path', 'file_name']:
                    v = img.get(key)
                    if v and os.path.splitext(os.path.basename(str(v)))[0] == orig_name:
                        orig_img_path = v
                        break
                if orig_img_path:
                    break
            
            if orig_img_path:
                models = get_models()
                if models['ap10k_model'] and models['device'] and models['visualizer']:
                    from app.dependencies import detect_and_visualize_keypoints
                    output_path, _ = detect_and_visualize_keypoints(
                        orig_img_path, models['ap10k_model'], models['device'], models['visualizer']
                    )
                    
                    if output_path and os.path.exists(output_path):
                        print(f"📷 (dynamic gen) 생성된 이미지 서빙: {output_path}")
                        return output_path
        
        except Exception as e:
            print(f"[dynamic gen] 예외 발생: {e}")
        
        return None
    
    def _create_dummy_image(self, filename: str) -> FileResponse:
        """더미 이미지 생성"""
        print(f"[dummy] 더미 이미지 생성: {filename}")
        
        if 'keypoint' in filename.lower():
            dummy_img = Image.new('RGB', (400, 400), color=(50, 50, 50))
            draw = ImageDraw.Draw(dummy_img)
            
            # 더미 키포인트 그리기
            keypoints = [(100, 100), (150, 80), (200, 120), (180, 200), (120, 220)]
            for kp in keypoints:
                draw.ellipse([kp[0]-5, kp[1]-5, kp[0]+5, kp[1]+5], fill='red')
            
            # 연결선 그리기
            connections = [(0,1), (1,2), (2,3), (3,4)]
            for conn in connections:
                draw.line([keypoints[conn[0]], keypoints[conn[1]]], fill='yellow', width=2)
        else:
            dummy_img = Image.new('RGB', (224, 224), color=(150, 100, 50))
            draw = ImageDraw.Draw(dummy_img)
            draw.text((50, 100), "강아지 이미지", fill='white')
        
        temp_path = os.path.join(settings.UPLOAD_FOLDER, f"temp_{filename}.jpg")
        dummy_img.save(temp_path, 'JPEG')
        print(f"[dummy] 더미 이미지 저장: {temp_path}")
        
        return FileResponse(temp_path, media_type="image/jpeg")
    
    def normalize_filename(self, filename: str) -> str:
        """파일명 정규화"""
        try:
            from training.visualize_keypoints import normalize_filename
            return normalize_filename(filename)
        except ImportError:
            import re
            name, ext = os.path.splitext(filename)
            name = re.sub(r'[^\w\d가-힣]+', '_', name)
            name = re.sub(r'_+', '_', name)
            name = name.strip('_')
            return name + ext
