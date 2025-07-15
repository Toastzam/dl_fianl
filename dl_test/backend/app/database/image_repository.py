"""
이미지 관련 데이터베이스 리포지토리
"""
import json
import os
import numpy as np
from typing import List, Dict, Optional
from mysql.connector import Error

from .connection import get_shared_db_connection, convert_datetime_to_string

class ImageRepository:
    """이미지 정보 관리 리포지토리"""
    
    def __init__(self):
        self.db_connection = get_shared_db_connection()
    
    def get_all_pet_images(self) -> List[Dict]:
        """SimCLR 유사도 검색용: 모든 펫 이미지 정보 반환"""
        images = self.get_all_pet_images_with_vectors()
        
        if not images:
            print("⚠️  벡터가 있는 이미지가 없습니다")
            return []
        
        parsed_images = []
        for img in images:
            try:
                # image_vector 파싱
                vector_json = img.get('image_vector')
                if vector_json:
                    if isinstance(vector_json, str):
                        vector = np.array(json.loads(vector_json), dtype=np.float32)
                    elif isinstance(vector_json, (list, np.ndarray)):
                        vector = np.array(vector_json, dtype=np.float32)
                    else:
                        print(f"⚠️  알 수 없는 벡터 형식: {type(vector_json)}")
                        continue
                else:
                    continue
                
                # 벡터 유효성 검사
                if vector.size == 0 or not np.isfinite(vector).all():
                    print(f"⚠️  무효한 벡터: {img.get('public_url')}")
                    continue
                
                # 결과에 추가
                img_copy = img.copy()
                img_copy['image_vector'] = vector
                img_copy = convert_datetime_to_string(img_copy)
                parsed_images.append(img_copy)
                
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"⚠️  벡터 파싱 실패: {img.get('public_url')} - {e}")
                continue
        
        print(f"✅ 파싱된 이미지 벡터: {len(parsed_images)}개")
        return parsed_images
    
    def get_all_pet_images_with_vectors(self) -> List[Dict]:
        """벡터가 있는 모든 펫 이미지 + 프로필 + 보호소 정보 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        pi.pet_image_id,
                        pi.pet_uid,
                        pi.file_name,
                        pi.public_url,
                        pi.image_vector,
                        pp.name,
                        pp.breed_cd as breed,
                        pp.weight_kg as weight,
                        pp.color,
                        pp.feature as description,
                        pp.found_location as location,
                        pp.adoption_status_cd as adoption_status,
                        pp.gender_cd as gender,
                        pp.birth_yyyy_mm as age,
                        pp.neutered_cd as neutered,
                        pp.reception_date,
                        pp.notice_start_date,
                        pp.notice_end_date,
                        pp.created_at,
                        pp.updated_at,
                        ash.shelter_name,
                        ash.shelter_phone,
                        ash.shelter_road_addr as shelter_address
                    FROM pet_image pi
                    JOIN pet_profile pp ON pi.pet_uid = pp.pet_uid
                    LEFT JOIN animal_shelter ash ON pp.shelter_id = ash.shelter_id
                    WHERE pi.image_vector IS NOT NULL 
                    AND pi.image_vector != ''
                """)
                images = cursor.fetchall()
                print(f"✅ 벡터+프로필+보호소 포함 {len(images)}개 이미지 조회")
                return images
        except Error as e:
            print(f"❌ 펫 이미지 벡터+프로필 조회 실패: {e}")
            return []
    
    def add_image_mapping(self, dog_id: int, image_path: str, feature_vector: List[float] = None) -> Optional[int]:
        """이미지 검색 기록 추가"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                
                feature_vector_json = json.dumps(feature_vector) if feature_vector else None
                
                cursor.execute("""
                    INSERT INTO pet_image_search_history (
                        image_filename, image_path, image_vector, created_at
                    ) VALUES (%s, %s, %s, NOW())
                """, (
                    os.path.basename(image_path) if image_path else None,
                    image_path, 
                    feature_vector_json
                ))
                
                conn.commit()
                search_id = cursor.lastrowid
                print(f"✅ 이미지 검색 기록 추가 성공 (search_id: {search_id})")
                return search_id
        except Error as e:
            print(f"❌ 이미지 검색 기록 추가 실패: {e}")
            return None
