"""
강아지 정보 관리 비즈니스 로직
"""
from typing import List, Dict, Any, Optional
from app.database import (
    get_all_dogs, get_dog_by_id, add_dog, update_dog, delete_dog,
    get_dog_by_image_path, add_image_mapping, get_breed_codes, get_breed_name_by_code
)

class DogService:
    def __init__(self):
        pass
    
    def get_all_dogs(self) -> List[Dict[str, Any]]:
        """모든 강아지 정보 조회"""
        dogs = get_all_dogs()
        return self.convert_breed_codes_in_dog_data(dogs)
    
    def get_dog_by_id(self, dog_id: int) -> Optional[Dict[str, Any]]:
        """특정 강아지 정보 조회"""
        dog = get_dog_by_id(dog_id)
        if dog:
            return self.convert_breed_codes_in_dog_data(dog)
        return None
    
    def create_dog(self, dog_data: Dict[str, Any]) -> int:
        """새 강아지 정보 추가"""
        return add_dog(dog_data)
    
    def update_dog(self, dog_id: int, dog_data: Dict[str, Any]) -> bool:
        """강아지 정보 수정"""
        return update_dog(dog_id, dog_data)
    
    def delete_dog(self, dog_id: int) -> bool:
        """강아지 정보 삭제"""
        return delete_dog(dog_id)
    
    def add_image_mapping(self, dog_id: int, image_path: str, feature_vector: List[float] = None) -> int:
        """강아지-이미지 매핑 추가"""
        return add_image_mapping(dog_id, image_path, feature_vector)
    
    def get_breed_codes(self) -> List[Dict[str, Any]]:
        """견종 코드 목록 조회"""
        return get_breed_codes()
    
    def convert_breed_codes_in_dog_data(self, dog_data):
        """강아지 데이터에서 견종 코드를 견종명으로 변환"""
        if isinstance(dog_data, list):
            return [self.convert_breed_codes_in_dog_data(dog) for dog in dog_data]
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
    
    def search_with_db_mapping(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과를 실제 DB 정보로 매핑"""
        mapped_results = []
        
        for i, result in enumerate(search_results):
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
                # DB에서 찾지 못한 경우 임시 정보 생성
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
        
        return mapped_results
