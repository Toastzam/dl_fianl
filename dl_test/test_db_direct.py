import sys
import os
sys.path.append('backend')

from database import get_all_dogs, get_breed_name_by_code

# DB에서 직접 데이터 가져오기
try:
    dogs = get_all_dogs()
    if dogs:
        first_dog = dogs[0]
        
        print("=== 첫 번째 강아지 정보 (DB 직접 조회) ===")
        print(f"ID: {first_dog.get('id')}")
        print(f"이름: {first_dog.get('name')}")
        print(f"견종코드: {first_dog.get('breed')}")  # DB에서는 breed_cd as breed
        print(f"성별코드: {first_dog.get('gender')}")  # DB에서는 gender_cd as gender
        print(f"입양상태코드: {first_dog.get('adoption_status')}")  # DB에서는 adoption_status_cd as adoption_status
        print()
        
        # 견종 코드 변환 테스트
        breed_code = first_dog.get('breed')
        if breed_code:
            breed_name = get_breed_name_by_code(breed_code)
            print(f"견종 변환 테스트: {breed_code} -> {breed_name}")
        
        print("\n=== 모든 키 ===")
        for key in sorted(first_dog.keys()):
            print(f"{key}: {first_dog[key]}")
            
except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()
