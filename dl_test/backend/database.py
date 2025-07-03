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
import numpy as np

# MySQL DB 연결 정보는 .env 파일에서 불러옵니다.
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3370)),  # 기본값을 3370으로 변경
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': 'utf8mb4',
    'autocommit': True,
    'use_pure': True,
    'connection_timeout': 10,
    'raise_on_warnings': True,
    # 'allow_public_key_retrieval': True,  # mysql-connector-python에서는 지원하지 않으므로 제거
    'use_unicode': True,
    'ssl_disabled': True  # 외부 NAS 접속 시 SSL 미사용(환경에 따라 조정)
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
        """MySQL DB 연결 생성 (TCP/IP 강제, named pipe 방지, None 안전 처리)"""
        try:
            db_config = self.db_config.copy()
            host_value = db_config.get('host')
            if not host_value:
                db_config['host'] = '127.0.0.1'
            elif str(host_value).strip() in ['localhost', '.', '::1']:
                db_config['host'] = '127.0.0.1'
            connection = mysql.connector.connect(**db_config)
            return connection
        except Error as e:
            print(f"❌ DB 연결 오류: {e}")
            print(f"[DEBUG] DB_CONFIG: {self.db_config}")
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
                    # print(f"✅ 기존 테이블 발견: {tables}")
                    pass
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
        """ID로 특정 강아지 정보 조회 (실제 pet_profile 테이블 사용) + breed_name 변환 및 누락 필드 보장"""
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
                    # breed_name 추가, 누락 필드 보장
                    dog = convert_datetime_to_string(dog)
                    dog['breed_name'] = self.get_breed_name_by_code(dog.get('breed')) if dog.get('breed') else None
                    for key in ['id','name','breed','breed_name','gender','neutered','weight','color','adoption_status','feature','location','age','reception_date','notice_start_date','notice_end_date','created_at','updated_at','shelter_name','shelter_phone','shelter_address','image_url']:
                        if key not in dog:
                            dog[key] = None
                    return dog
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
                # print(f"📋 데이터베이스 테이블 목록: {tables}")
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
                # print(f"📋 테이블 '{table_name}' 구조:")
                for col in columns:
                    # print(f"  - {col['Field']}: {col['Type']} {'(Primary Key)' if col['Key'] == 'PRI' else ''}")
                    pass
                return columns
        except Error as e:
            print(f"❌ 테이블 '{table_name}' 구조 조회 실패: {e}")
            return []
    
    def get_all_pet_images_with_vectors(self) -> List[Dict]:
        """
        벡터가 있는 모든 펫 이미지 + 프로필 + 보호소 정보까지 한 번에 조인해서 반환 (속도 개선)
        """
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
                print(f"✅ 벡터+프로필+보호소 포함 {len(images)}개 이미지 조회 (1회 쿼리)!")
                return images
        except Error as e:
            print(f"❌ 펫 이미지 벡터+프로필 조회 실패: {e}")
            return []

    def get_breed_name_by_code(self, breed_code: str) -> str:
        """견종 코드로 견종 이름 조회 (클래스 메서드 버전)"""
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

    def get_breed_codes(self) -> Dict[str, str]:
        """cmn_code 테이블에서 DOG_BREED 코드와 이름 매핑 조회 (클래스 메서드 버전)"""
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
                breed_dict = {row['cd']: row['cd_nm'] for row in breed_codes}
                print(f"✅ DOG_BREED 코드 {len(breed_dict)}개 조회 (클래스)")
                return breed_dict
        except Error as e:
            print(f"❌ DOG_BREED 코드 조회 실패: {e}")
            return {}

    def get_all_pet_images(self) -> List[Dict]:
        """
        SimCLR 유사도 검색용: public_url, image_vector, 그리고 모든 강아지 정보(프론트엔드 요구 필드 포함) 반환 (robust, 빠름)
        - public_url, image_vector, 실제 파일 존재, 벡터 파싱, 견종명 변환 캐시 등 robust하게 처리
        """
        images = self.get_all_pet_images_with_vectors()
        print(f"[DEBUG] 1. get_all_pet_images_with_vectors() 반환: {len(images)}개")
        # 견종명 캐시 (한 번만 쿼리)
        breed_dict = self.get_breed_codes()  # {code: name}
        print(f"[DEBUG] 2. get_breed_codes() 반환: {len(breed_dict)}개")
        result = []
        for idx, img in enumerate(images):
            # print(f"[DEBUG] 3.{idx+1} 원본 img: {json.dumps({k: (str(v)[:80] if k=='image_vector' else v) for k,v in img.items()}, ensure_ascii=False, default=str)}") if idx < 3 else None
            public_url = img.get('public_url')
            if not public_url or not isinstance(public_url, str) or public_url.strip() == '':
                print(f"[SKIP] public_url 누락: {img}")
                continue
            # 외부 URL(http/https)도 허용
            file_found = False
            if public_url.startswith('http://') or public_url.startswith('https://'):
                file_found = True
            else:
                for folder in ['uploads', 'static', 'output_keypoints']:
                    file_path = os.path.join(folder, os.path.basename(public_url))
                    if os.path.exists(file_path):
                        file_found = True
                        break
            if not file_found:
                print(f"[SKIP] 파일 미존재: {public_url}")
                continue
            vec = img.get('image_vector')
            # image_vector가 JSON 문자열, bytes, 리스트 등 다양한 형태일 수 있음
            try:
                if isinstance(vec, str):
                    vec = np.array(json.loads(vec), dtype=np.float32)
                elif isinstance(vec, (bytes, bytearray)):
                    vec = np.frombuffer(vec, dtype=np.float32)
                elif isinstance(vec, list):
                    vec = np.array(vec, dtype=np.float32)
                # print(f"[DEBUG] 4.{idx+1} image_vector 파싱 성공: shape={vec.shape if hasattr(vec, 'shape') else type(vec)}")  # 과도한 디버그 출력 주석 처리
            except Exception as e:
                print(f"[SKIP] image_vector 파싱 실패: {public_url}, error: {e}")
                continue
            if vec is None or not hasattr(vec, 'shape') or vec.shape[0] < 10:
                print(f"[SKIP] image_vector None/짧음: {public_url}")
                continue
            breed_code = img.get('breed')
            breed_name = breed_dict.get(breed_code) if breed_code else None
            merged = {
                'public_url': public_url,
                'image_url': public_url,  # 항상 image_url도 포함
                'image_vector': vec,
                'pet_uid': img.get('pet_uid'),
                'file_name': img.get('file_name'),
                'id': img.get('pet_uid'),
                'name': img.get('name'),
                'breed': breed_code,
                'breed_name': breed_name,
                'gender': img.get('gender'),
                'neutered': img.get('neutered'),
                'weight': img.get('weight'),
                'color': img.get('color'),
                'adoption_status': img.get('adoption_status'),
                'feature': img.get('description'),
                'location': img.get('location'),
                'age': img.get('age'),
                'reception_date': img.get('reception_date'),
                'notice_start_date': img.get('notice_start_date'),
                'notice_end_date': img.get('notice_end_date'),
                'created_at': img.get('created_at'),
                'updated_at': img.get('updated_at'),
                'shelter_name': img.get('shelter_name'),
                'shelter_phone': img.get('shelter_phone'),
                'shelter_address': img.get('shelter_address'),
            }
            # print(f"[DEBUG] 5.{idx+1} merged dict: {json.dumps({k: (str(v)[:80] if k=='image_vector' else v) for k,v in merged.items()}, ensure_ascii=False, default=str)}") if idx < 3 else None
            result.append(merged)
        print(f"[INFO] 최종 사용 가능한 이미지 {len(result)}개")
        # print(f"[DEBUG] 최종 results: {results}")
        return convert_datetime_to_string(result)

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

def get_all_pet_images():
    return db.get_all_pet_images()

def get_dog_by_id(dog_id: int):
    return db.get_dog_by_id(dog_id)
