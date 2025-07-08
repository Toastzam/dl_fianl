"""
실제 DB 데이터 조회 테스트
"""

from database import DogDatabase

def test_database():
    print("🧪 실제 DB 데이터 조회 테스트")
    
    try:
        db = DogDatabase()
        
        # 1. 전체 강아지 수 확인
        dogs = db.get_all_dogs()
        print(f"📊 전체 강아지 수: {len(dogs)}")
        
        if dogs:
            # 첫 번째 강아지 정보 출력
            first_dog = dogs[0]
            print(f"\n🐕 첫 번째 강아지 정보:")
            for key, value in first_dog.items():
                print(f"  {key}: {value}")
        
        # 2. 벡터가 있는 이미지 수 확인
        images = db.get_all_pet_images_with_vectors()
        print(f"\n📊 벡터가 있는 이미지 수: {len(images)}")
        
        if images:
            # 첫 번째 이미지 정보 출력 (벡터 제외)
            first_image = images[0]
            print(f"\n🖼️ 첫 번째 이미지 정보:")
            for key, value in first_image.items():
                if key != 'image_vector':  # 벡터는 너무 길어서 제외
                    print(f"  {key}: {value}")
                else:
                    vector_length = len(value) if value else 0
                    print(f"  {key}: (벡터 길이: {vector_length})")
        
        # 3. 특정 ID로 조회 테스트
        if dogs:
            test_id = dogs[0]['id']
            dog_detail = db.get_dog_by_id(test_id)
            print(f"\n🔍 ID {test_id} 강아지 상세 정보:")
            if dog_detail:
                for key, value in dog_detail.items():
                    if key != 'image_vector':
                        print(f"  {key}: {value}")
        
        print(f"\n✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_database()
