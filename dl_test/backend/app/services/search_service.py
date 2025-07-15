"""
검색 관련 비즈니스 로직
"""
import os
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw

from app.config import settings
from app.dependencies import get_models, get_feature_service
from app.database import get_all_pet_images

class SearchService:
    def __init__(self):
        self.feature_service = get_feature_service()
    
    async def real_model_search(self, file_location: str, filename: str) -> Dict[str, Any]:
        """실제 모델을 사용한 검색"""
        start_time = time.time()
        models = get_models()
        
        try:
            print("🔍 실제 모델 사용 - 시작 (DB public_url 기반)")
            
            # 1. 쿼리 이미지 키포인트 검출
            from app.dependencies import detect_and_visualize_keypoints
            query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
                file_location, models['ap10k_model'], models['device'], models['visualizer']
            )
            
            # 강아지 판별 로직
            is_dog_by_kp, is_dog_by_simclr, dog_check_info = await self._check_if_dog(
                query_pose_results, file_location
            )
            
            if not (is_dog_by_kp and is_dog_by_simclr):
                return self._create_not_dog_response(
                    file_location, query_kp_output_path, dog_check_info
                )
            
            # 2. SimCLR 기반 유사 이미지 검색
            similar_results = await self._simclr_search(file_location)
            
            # 3. 키포인트 유사도 계산 및 종합
            results = await self._combine_similarities(
                similar_results, query_pose_results, models
            )
            
            processing_time = time.time() - start_time
            
            return self._create_success_response(
                file_location, query_kp_output_path, results, processing_time, 'real_model_db_public_url'
            )
            
        except Exception as e:
            print(f"❌ 실제 모델 검색 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return await self.dummy_search(file_location, filename)
    
    async def dummy_search(self, file_location: str, filename: str) -> Dict[str, Any]:
        """더미 데이터를 사용한 검색"""
        print("🔄 더미 모드 사용 - 실제 DB에서 랜덤 샘플링")
        
        start_time = time.time()
        
        try:
            from app.database import get_all_dogs
            total_dogs = len(get_all_dogs())
            
            if total_dogs == 0:
                return self._create_fallback_dummy_response(file_location)
            
            # 더미 결과 생성
            dummy_results = self._generate_dummy_results(total_dogs)
            processing_time = time.time() - start_time
            
            return self._create_success_response(
                file_location, None, dummy_results, processing_time, 'dummy_with_real_db'
            )
            
        except Exception as e:
            print(f"❌ 더미 검색 실패: {e}")
            return self._create_fallback_dummy_response(file_location)
    
    async def _check_if_dog(self, query_pose_results, file_location: str) -> tuple:
        """강아지 판별 로직"""
        # 키포인트 기반 판별
        keypoints_info = self._extract_keypoints_info(query_pose_results)
        is_dog_by_kp = (
            keypoints_info['num_keypoints'] >= settings.MIN_KEYPOINTS and 
            keypoints_info['avg_score'] >= settings.MIN_AVG_SCORE
        )
        
        # SimCLR 기반 판별
        pet_images = get_all_pet_images()
        if not pet_images:
            return False, False, keypoints_info
        
        query_vector = self.feature_service.extract_features_from_path(file_location)
        db_vectors = np.stack([img['image_vector'] for img in pet_images])
        similarities = np.dot(db_vectors, query_vector) / (
            np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )
        
        max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        is_dog_by_simclr = max_sim >= settings.SIMCLR_MIN_SIM
        
        dog_check_info = {
            **keypoints_info,
            'max_simclr_similarity': max_sim,
            'is_dog_by_kp': is_dog_by_kp,
            'is_dog_by_simclr': is_dog_by_simclr
        }
        
        return is_dog_by_kp, is_dog_by_simclr, dog_check_info
    
    def _extract_keypoints_info(self, query_pose_results) -> Dict[str, Any]:
        """키포인트 정보 추출"""
        if isinstance(query_pose_results, list) and len(query_pose_results) > 0:
            kp_data = query_pose_results[0] if isinstance(query_pose_results[0], dict) else {}
        elif isinstance(query_pose_results, dict):
            kp_data = query_pose_results
        else:
            kp_data = {}
        
        keypoints = kp_data.get('keypoints', [])
        if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 3:
            scores = keypoints[:, 2]
            num_keypoints = len(keypoints)
            avg_score = float(np.mean(scores)) if scores.size > 0 else 0.0
        else:
            num_keypoints = 0
            avg_score = 0.0
        
        return {
            'num_keypoints': num_keypoints,
            'avg_score': avg_score
        }
    
    async def _simclr_search(self, file_location: str) -> List[Dict[str, Any]]:
        """SimCLR 기반 검색"""
        pet_images = get_all_pet_images()
        if not pet_images:
            raise Exception("DB에 등록된 강아지 이미지가 없습니다.")
        
        query_vector = self.feature_service.extract_features_from_path(file_location)
        db_vectors = np.stack([img['image_vector'] for img in pet_images])
        similarities = np.dot(db_vectors, query_vector) / (
            np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )
        
        # top_k 추출
        top_k = 6
        valid_indices = [i for i, s in enumerate(similarities) if np.isfinite(s)]
        similarities_valid = similarities[valid_indices]
        
        sorted_indices = np.argsort(similarities_valid)[::-1][:top_k]
        top_indices = [valid_indices[i] for i in sorted_indices]
        
        results = []
        for idx in top_indices:
            simclr_score = float(similarities[idx])
            db_img = pet_images[idx].copy()
            if 'image_vector' in db_img:
                del db_img['image_vector']
            
            db_image_url = self._process_image_url(db_img)
            
            results.append({
                'similarity': simclr_score,
                'image_url': db_image_url,
                'db_info': db_img
            })
        
        return results
    
    def _process_image_url(self, db_img: Dict[str, Any]) -> Optional[str]:
        """이미지 URL 처리"""
        db_image_url = db_img.get('public_url') or db_img.get('image_url') or db_img.get('image_path')
        
        if not db_image_url:
            return None
        
        # 외부 URL은 그대로 사용
        if str(db_image_url).startswith(('http://', 'https://')):
            return db_image_url
        
        # 로컬 파일 처리
        base_name = os.path.basename(str(db_image_url))
        uploads_path = os.path.join(settings.UPLOAD_FOLDER, base_name)
        
        if os.path.exists(uploads_path):
            return f"/uploads/{base_name}"
        
        return db_image_url
    
    async def _combine_similarities(self, similar_results: List[Dict], query_pose_results, models) -> List[Dict]:
        """SimCLR과 키포인트 유사도 결합"""
        from app.dependencies import detect_and_visualize_keypoints, calculate_keypoint_similarity
        
        results = []
        for i, sim_result in enumerate(similar_results):
            simclr_score = sim_result.get('similarity', 0.0)
            image_url = sim_result.get('image_url')
            db_info = sim_result.get('db_info', {}).copy()
            
            if 'image_vector' in db_info:
                del db_info['image_vector']
            
            keypoint_similarity = 0.0
            similar_kp_output_path = None
            keypoint_time = None
            
            try:
                kp_start = time.time()
                similar_kp_output_path, similar_pose_results = detect_and_visualize_keypoints(
                    image_url, models['ap10k_model'], models['device'], models['visualizer']
                )
                
                if query_pose_results and similar_pose_results:
                    keypoint_similarity = calculate_keypoint_similarity(
                        query_pose_results, similar_pose_results
                    )
                
                keypoint_time = round(time.time() - kp_start, 3)
            except Exception as e:
                print(f"⚠️ 키포인트 검출 실패 (URL: {image_url}): {e}")
            
            combined_similarity = (
                settings.SIMCLR_WEIGHT * simclr_score + 
                settings.KEYPOINT_WEIGHT * keypoint_similarity
            )
            
            result_dict = {
                'rank': i + 1,
                'image_url': image_url,
                'keypoint_image_path': self._to_output_keypoints_url(similar_kp_output_path),
                'simclr_similarity': float(simclr_score),
                'keypoint_similarity': float(keypoint_similarity),
                'combined_similarity': float(combined_similarity),
                'db_info': db_info,
                'keypoint_processing_time': keypoint_time
            }
            results.append(result_dict)
        
        # 복합 유사도로 정렬
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def _to_output_keypoints_url(self, path: Optional[str]) -> Optional[str]:
        """output_keypoints URL 변환"""
        if not path:
            return None
        return f"/output_keypoints/{os.path.basename(path)}"
    
    def _generate_dummy_results(self, total_dogs: int) -> List[Dict[str, Any]]:
        """더미 검색 결과 생성"""
        dummy_results = []
        for i in range(6):
            fake_id = random.randint(1, min(total_dogs, 1000))
            simclr_sim = random.uniform(0.7, 0.95)
            keypoint_sim = random.uniform(0.6, 0.9)
            combined_sim = (settings.SIMCLR_WEIGHT * simclr_sim) + (settings.KEYPOINT_WEIGHT * keypoint_sim)
            
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
        
        dummy_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        for i, result in enumerate(dummy_results):
            result['rank'] = i + 1
        
        return dummy_results
    
    def _create_not_dog_response(self, file_location: str, query_kp_output_path: str, dog_check_info: Dict) -> Dict:
        """강아지가 아닌 경우 응답"""
        reason = []
        if not dog_check_info['is_dog_by_kp']:
            reason.append(f"키포인트 개수/신뢰도 부족 (개수: {dog_check_info['num_keypoints']}, 평균: {dog_check_info['avg_score']:.2f})")
        if not dog_check_info['is_dog_by_simclr']:
            reason.append(f"SimCLR DB 유사도 낮음 (최대: {dog_check_info['max_simclr_similarity']:.2f})")
        
        msg = "업로드된 이미지는 강아지로 인식되지 않습니다. " + ", ".join(reason)
        
        return {
            'success': False,
            'error': 'not_a_dog',
            'message': msg,
            'query_image': file_location.replace('\\', '/'),
            'query_keypoint_image': self._to_output_keypoints_url(query_kp_output_path),
            'dog_check': dog_check_info
        }
    
    def _create_success_response(self, file_location: str, query_kp_output_path: str, 
                               results: List[Dict], processing_time: float, mode: str) -> Dict:
        """성공 응답 생성"""
        try:
            from app.database import get_all_dogs
            total_dogs = len(get_all_dogs())
        except:
            total_dogs = 10000
        
        return {
            'success': True,
            'query_image': '/uploads/' + os.path.basename(file_location),
            'query_keypoint_image': self._to_output_keypoints_url(query_kp_output_path),
            'results': results,
            'mode': mode,
            'search_metadata': {
                'database_size': total_dogs,
                'images_with_data': total_dogs,
                'searched_results': len(results),
                'confidence_threshold': 0.60,
                'algorithm': 'SimCLR + AP-10K Hybrid AI',
                'processing_time': round(processing_time, 2),
                'model_version': settings.SIMCLR_MODEL_VERSION,
                'feature_dimension': 2048
            }
        }
    
    def _create_fallback_dummy_response(self, file_location: str) -> Dict:
        """완전 폴백 더미 응답"""
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
        
        return {
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
        }
