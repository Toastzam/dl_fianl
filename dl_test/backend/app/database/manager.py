"""
데이터베이스 관리자 (기존 호환성 유지)
"""
from typing import List, Dict, Optional
from mysql.connector import Error

from .connection import get_shared_db_connection
from .dog_repository import DogRepository
from .image_repository import ImageRepository
from .code_repository import CodeRepository

class DatabaseManager:
    """데이터베이스 통합 관리자"""
    
    def __init__(self):
        self.connection = get_shared_db_connection()
        self.dog_repo = DogRepository()
        self.image_repo = ImageRepository()
        self.code_repo = CodeRepository()
    
    def show_tables(self) -> List[str]:
        """테이블 목록 조회"""
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                return tables
        except Error as e:
            print(f"❌ 테이블 목록 조회 실패: {e}")
            return []
    
    def describe_table(self, table_name: str) -> List[Dict]:
        """테이블 구조 조회"""
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                return columns
        except Error as e:
            print(f"❌ 테이블 '{table_name}' 구조 조회 실패: {e}")
            return []
    
    # Dog Repository 메서드들
    def get_all_dogs(self) -> List[Dict]:
        return self.dog_repo.get_all_dogs()
    
    def get_dog_by_id(self, dog_id: int) -> Optional[Dict]:
        return self.dog_repo.get_dog_by_id(dog_id)
    
    def add_dog(self, dog_data: Dict) -> int:
        return self.dog_repo.add_dog(dog_data)
    
    def update_dog(self, dog_id: int, dog_data: Dict) -> bool:
        return self.dog_repo.update_dog(dog_id, dog_data)
    
    def delete_dog(self, dog_id: int) -> bool:
        return self.dog_repo.delete_dog(dog_id)
    
    def get_dog_by_image_path(self, image_path: str) -> Optional[Dict]:
        return self.dog_repo.get_dog_by_image_path(image_path)
    
    def search_dogs(self, **criteria) -> List[Dict]:
        return self.dog_repo.search_dogs(**criteria)
    
    # Image Repository 메서드들
    def get_all_pet_images(self) -> List[Dict]:
        return self.image_repo.get_all_pet_images()
    
    def add_image_mapping(self, dog_id: int, image_path: str, feature_vector: List[float] = None):
        return self.image_repo.add_image_mapping(dog_id, image_path, feature_vector)
    
    # Code Repository 메서드들
    def get_breed_codes(self) -> List[Dict[str, str]]:
        return self.code_repo.get_breed_codes()
    
    def get_breed_name_by_code(self, breed_code: str) -> str:
        return self.code_repo.get_breed_name_by_code(breed_code)

# 기존 호환성을 위한 전역 인스턴스
db = DatabaseManager()

# 편의 함수들 (기존 코드 호환성)
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
