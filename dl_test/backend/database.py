"""
데이터베이스 연결 및 강아지 정보 관리 모듈
MySQL을 사용한 DB 구현
"""

import mysql.connector
from mysql.connector import Error
import os
from typing import List, Dict, Optional
import json
from datetime import datetime, date
from decimal import Decimal

# MySQL DB 연결 정보
DB_CONFIG = {
    'host': 'byhou.synology.me',
    'port': 3370,
    'user': 'h3',
    'password': 'Dbrlrus25^',
    'database': 'h3',
    'charset': 'utf8mb4',
    'autocommit': True
}

def serialize_datetime(obj):
    """datetime 객체를 JSON 직렬화 가능한 문자열로 변환"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj

def convert_datetime_to_string(data):
    """딕셔너리나 리스트의 datetime 객체들을 문자열로 변환"""
    if isinstance(data, dict):
        return {key: convert_datetime_to_string(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_datetime_to_string(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat() if data else None
    elif isinstance(data, Decimal):
        return float(data) if data is not None else None
    else:
        return data

class DogDatabase:
    def __init__(self):
        self.db_config = DB_CONFIG
        # 연결 테스트 및 기존 테이블 확인
        try:
            conn = self.get_connection()
            if conn.is_connected():
                print(f"✅ MySQL 데이터베이스 연결 성공: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
                conn.close()
                # 기존 테이블 확인
                self.check_database_connection()
        except Error as e:
            print(f"❌ MySQL 연결 실패: {e}")
    
    def get_connection(self):
        """MySQL DB 연결 생성"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            print(f"❌ DB 연결 오류: {e}")
            raise
    
    def check_database_connection(self):
        """데이터베이스 연결 및 기존 테이블 확인"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 기존 테이블 목록 확인
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                
                if tables:
                    print(f"✅ 기존 테이블 발견: {tables}")
                else:
                    print("⚠️  데이터베이스에 테이블이 없습니다.")
                
                return tables
                
        except Error as e:
            print(f"❌ 데이터베이스 확인 실패: {e}")
            raise
    
    def add_dog(self, dog_data: Dict) -> int:
        """새 강아지 정보 추가 - 실제 테이블 구조에 맞게 수정 필요"""
        print("⚠️  add_dog: 실제 테이블 구조 확인 후 구현 필요")
        return 0
        # TODO: 실제 테이블명과 컬럼명 확인 후 구현
    
    def get_all_dogs(self) -> List[Dict]:
        """모든 강아지 정보 조회 (실제 pet_profile 테이블 사용)"""
        try:
            with self.get_connection() as conn:
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
                
                # datetime 객체를 문자열로 변환
                dogs_converted = [convert_datetime_to_string(dog) for dog in dogs]
                return dogs_converted
        except Error as e:
            print(f"❌ 강아지 목록 조회 실패: {e}")
            return []
    
    def get_dog_by_id(self, dog_id: int) -> Optional[Dict]:
        """ID로 특정 강아지 정보 조회 (실제 pet_profile 테이블 사용)"""
        try:
            with self.get_connection() as conn:
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
                    print(f"✅ pet_profile에서 강아지 ID {dog_id} 정보 조회 성공")
                    # datetime 객체를 문자열로 변환
                    return convert_datetime_to_string(dog)
                else:
                    print(f"⚠️  강아지 ID {dog_id}를 찾을 수 없습니다")
                return None
        except Error as e:
            print(f"❌ 강아지 조회 실패: {e}")
            return None
    
    def update_dog(self, dog_id: int, dog_data: Dict) -> bool:
        """강아지 정보 업데이트 - 실제 테이블 구조에 맞게 수정 필요"""
        print(f"⚠️  update_dog({dog_id}): 실제 테이블 구조 확인 후 구현 필요")
        return False
        # TODO: 실제 테이블명과 컬럼명 확인 후 구현
    
    def delete_dog(self, dog_id: int) -> bool:
        """강아지 정보 삭제 - 실제 테이블 구조에 맞게 수정 필요"""
        print(f"⚠️  delete_dog({dog_id}): 실제 테이블 구조 확인 후 구현 필요")
        return False
        # TODO: 실제 테이블명과 컬럼명 확인 후 구현
    
    def add_image_mapping(self, dog_id: int, image_path: str, feature_vector: List[float] = None):
        """이미지 검색 기록 추가 (pet_image_search_history 테이블 사용)"""
        try:
            with self.get_connection() as conn:
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
    
    def get_dog_by_image_path(self, image_path: str) -> Optional[Dict]:
        """이미지 경로로 강아지 정보 조회 (pet_image 테이블 사용)"""
        try:
            with self.get_connection() as conn:
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
        """강아지 검색 (실제 pet_profile 테이블 사용)"""
        try:
            with self.get_connection() as conn:
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
                
                cursor.execute(f"""
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
                        pp.created_at,
                        pi.public_url as image_url
                    FROM pet_profile pp
                    LEFT JOIN pet_image pi ON pp.pet_uid = pi.pet_uid
                    WHERE {where_clause}
                    ORDER BY pp.created_at DESC
                    LIMIT 50
                """, params)
                
                dogs = cursor.fetchall()
                print(f"✅ 검색 조건으로 {len(dogs)}개 강아지 정보 조회")
                return dogs
        except Error as e:
            print(f"❌ 강아지 검색 실패: {e}")
            return []
    
    def show_tables(self) -> List[str]:
        """데이터베이스의 모든 테이블 목록 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                print(f"📋 데이터베이스 테이블 목록: {tables}")
                return tables
        except Error as e:
            print(f"❌ 테이블 목록 조회 실패: {e}")
            return []
    
    def describe_table(self, table_name: str) -> List[Dict]:
        """특정 테이블의 구조 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print(f"📋 테이블 '{table_name}' 구조:")
                for col in columns:
                    print(f"  - {col['Field']}: {col['Type']} {'(Primary Key)' if col['Key'] == 'PRI' else ''}")
                return columns
        except Error as e:
            print(f"❌ 테이블 '{table_name}' 구조 조회 실패: {e}")
            return []
    
    def get_all_pet_images_with_vectors(self) -> List[Dict]:
        """벡터가 있는 모든 펫 이미지 조회 (유사도 검색용)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        pi.pet_image_id,
                        pi.pet_uid,
                        pi.file_name,
                        pi.public_url,
                        pi.image_vector,
                        pp.name,
                        pp.breed_cd,
                        pp.adoption_status_cd
                    FROM pet_image pi
                    JOIN pet_profile pp ON pi.pet_uid = pp.pet_uid
                    WHERE pi.image_vector IS NOT NULL 
                    AND pi.image_vector != ''
                """)
                
                images = cursor.fetchall()
                print(f"✅ 벡터가 있는 {len(images)}개 이미지 조회")
                return images
        except Error as e:
            print(f"❌ 펫 이미지 벡터 조회 실패: {e}")
            return []
    
    def save_search_result(self, search_id: int, pet_uid: int, pet_image_id: int, similarity: float):
        """검색 결과 저장 (pet_image_match_result 테이블)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pet_image_match_result (
                        search_id, pet_image_id, pet_uid, user_id, similarity, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                """, (search_id, pet_image_id, pet_uid, 1, similarity))  # user_id는 임시로 1
                
                conn.commit()
                print(f"✅ 검색 결과 저장: pet_uid={pet_uid}, similarity={similarity:.4f}")
                return cursor.lastrowid
        except Error as e:
            print(f"❌ 검색 결과 저장 실패: {e}")
            return None
    
    def get_breed_codes(self) -> Dict[str, str]:
        """cmn_code 테이블에서 DOG_BREED 코드와 이름 매핑 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT cd, cd_nm
                    FROM cmn_code
                    WHERE group_cd = 'DOG_BREED' AND use_yn = 'Y'
                    ORDER BY cd_nm
                """)
                
                breed_codes = cursor.fetchall()
                # 딕셔너리로 변환 (cd -> cd_nm)
                breed_dict = {row['cd']: row['cd_nm'] for row in breed_codes}
                print(f"✅ DOG_BREED 코드 {len(breed_dict)}개 조회")
                return breed_dict
        except Error as e:
            print(f"❌ DOG_BREED 코드 조회 실패: {e}")
            return {}

    def get_breed_name_by_code(self, breed_code: str) -> str:
        """견종 코드로 견종 이름 조회"""
        if not breed_code:
            return '정보 없음'
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cd_nm
                    FROM cmn_code
                    WHERE group_cd = 'DOG_BREED' AND cd = %s AND use_yn = 'Y'
                """, (breed_code,))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    return breed_code  # 코드를 찾을 수 없으면 원본 코드 반환
        except Error as e:
            print(f"❌ 견종 코드 '{breed_code}' 조회 실패: {e}")
            return breed_code

# 전역 DB 인스턴스
db = DogDatabase()

# DatabaseManager 별칭 (backward compatibility)
DatabaseManager = DogDatabase

# 편의 함수들
def get_all_dogs():
    return db.get_all_dogs()

def get_dog_by_id(dog_id: int):
    return db.get_dog_by_id(dog_id)

def add_dog(dog_data: Dict):
    return db.add_dog(dog_data)

def update_dog(dog_id: int, dog_data: Dict):
    return db.update_dog(dog_id, dog_data)

def delete_dog(dog_id: int):
    return db.delete_dog(dog_id)

def get_dog_by_image_path(image_path: str):
    return db.get_dog_by_image_path(image_path)

def add_image_mapping(dog_id: int, image_path: str, feature_vector: List[float] = None):
    return db.add_image_mapping(dog_id, image_path, feature_vector)

def search_dogs(**criteria):
    return db.search_dogs(**criteria)

def get_breed_codes():
    return db.get_breed_codes()

def get_breed_name_by_code(breed_code: str):
    return db.get_breed_name_by_code(breed_code)
