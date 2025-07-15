"""
강아지 관련 데이터베이스 리포지토리
"""
import os
import json
from typing import List, Dict, Optional
from mysql.connector import Error

from .connection import get_shared_db_connection, convert_datetime_to_string

class DogRepository:
    """강아지 정보 관리 리포지토리"""
    
    def __init__(self):
        self.db_connection = get_shared_db_connection()
    
    def get_all_dogs(self) -> List[Dict]:
        """모든 강아지 정보 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        pp.pet_uid as id,
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
                        pi.public_url as image_url
                    FROM pet_profile pp
                    LEFT JOIN pet_image pi ON pp.pet_uid = pi.pet_uid
                    ORDER BY pp.created_at DESC
                """)
                
                dogs = cursor.fetchall()
                print(f"✅ pet_profile에서 {len(dogs)}개 강아지 정보 조회")
                
                return [convert_datetime_to_string(dog) for dog in dogs]
        except Error as e:
            print(f"❌ 강아지 목록 조회 실패: {e}")
            return []
    
    def get_dog_by_id(self, dog_id: int) -> Optional[Dict]:
        """ID로 특정 강아지 정보 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        pp.pet_uid as id,
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
                        pi.public_url as image_url,
                        pi.image_vector,
                        ash.shelter_name,
                        ash.shelter_phone,
                        ash.shelter_road_addr as shelter_address
                    FROM pet_profile pp
                    LEFT JOIN pet_image pi ON pp.pet_uid = pi.pet_uid
                    LEFT JOIN animal_shelter ash ON pp.shelter_id = ash.shelter_id
                    WHERE pp.pet_uid = %s
                """, (dog_id,))
                
                dog = cursor.fetchone()
                if dog:
                    dog = convert_datetime_to_string(dog)
                    # 누락 필드 보장
                    required_fields = [
                        'id', 'name', 'breed', 'breed_name', 'gender', 'neutered', 
                        'weight', 'color', 'adoption_status', 'feature', 'location', 
                        'age', 'reception_date', 'notice_start_date', 'notice_end_date',
                        'created_at', 'updated_at', 'shelter_name', 'shelter_phone', 
                        'shelter_address', 'image_url'
                    ]
                    for key in required_fields:
                        if key not in dog:
                            dog[key] = None
                    return dog
                else:
                    print(f"⚠️  강아지 ID {dog_id}를 찾을 수 없습니다")
                return None
        except Error as e:
            print(f"❌ 강아지 조회 실패: {e}")
            return None
    
    def add_dog(self, dog_data: Dict) -> int:
        """새 강아지 정보 추가"""
        print("⚠️  add_dog: 실제 테이블 구조 확인 후 구현 필요")
        return 0
    
    def update_dog(self, dog_id: int, dog_data: Dict) -> bool:
        """강아지 정보 업데이트"""
        print(f"⚠️  update_dog({dog_id}): 실제 테이블 구조 확인 후 구현 필요")
        return False
    
    def delete_dog(self, dog_id: int) -> bool:
        """강아지 정보 삭제"""
        print(f"⚠️  delete_dog({dog_id}): 실제 테이블 구조 확인 후 구현 필요")
        return False
    
    def get_dog_by_image_path(self, image_path: str) -> Optional[Dict]:
        """이미지 경로로 강아지 정보 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        pp.pet_uid as id,
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
                        pp.created_at,
                        pi.public_url as image_url,
                        pi.image_vector
                    FROM pet_profile pp
                    JOIN pet_image pi ON pp.pet_uid = pi.pet_uid
                    WHERE pi.public_url = %s OR pi.file_name = %s
                """, (image_path, os.path.basename(image_path) if image_path else None))
                
                dog = cursor.fetchone()
                if dog:
                    print(f"✅ 이미지 경로로 강아지 정보 조회 성공")
                else:
                    print(f"⚠️  이미지 경로 '{image_path}'로 강아지를 찾을 수 없습니다")
                return dog
        except Error as e:
            print(f"❌ 이미지 경로로 강아지 조회 실패: {e}")
            return None
    
    def search_dogs(self, **criteria) -> List[Dict]:
        """강아지 검색"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                conditions = []
                params = []
                
                # 검색 조건 매핑
                if criteria.get('breed'):
                    conditions.append("pp.breed_cd LIKE %s")
                    params.append(f"%{criteria['breed']}%")
                
                if criteria.get('gender'):
                    conditions.append("pp.gender_cd = %s")
                    params.append(criteria['gender'])
                
                if criteria.get('location'):
                    conditions.append("pp.found_location LIKE %s")
                    params.append(f"%{criteria['location']}%")
                
                if criteria.get('adoption_status'):
                    conditions.append("pp.adoption_status_cd = %s")
                    params.append(criteria['adoption_status'])
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                query = f"""
                    SELECT 
                        pp.pet_uid as id,
                        pp.name,
                        pp.breed_cd as breed,
                        pp.weight_kg as weight,
                        pp.color,
                        pp.feature as description,
                        pp.found_location as location,
                        pp.adoption_status_cd as adoption_status,
                        pp.gender_cd as gender,
                        pp.birth_yyyy_mm as age,
                        pp.created_at,
                        pi.public_url as image_url
                    FROM pet_profile pp
                    LEFT JOIN pet_image pi ON pp.pet_uid = pi.pet_uid
                    WHERE {where_clause}
                    ORDER BY pp.created_at DESC
                """
                
                cursor.execute(query, params)
                dogs = cursor.fetchall()
                
                print(f"✅ 검색 조건으로 {len(dogs)}개 강아지 조회")
                return [convert_datetime_to_string(dog) for dog in dogs]
        except Error as e:
            print(f"❌ 강아지 검색 실패: {e}")
            return []
