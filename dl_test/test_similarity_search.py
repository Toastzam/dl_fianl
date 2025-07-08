#!/usr/bin/env python3
import requests
import json

# 더미 이미지 파일 생성 (테스트용)
from PIL import Image
import io

# 더미 이미지 생성
dummy_img = Image.new('RGB', (224, 224), color=(150, 100, 50))
img_bytes = io.BytesIO()
dummy_img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# 유사도 검색 API 호출
try:
    files = {'file': ('test_dog.jpg', img_bytes, 'image/jpeg')}
    response = requests.post('http://localhost:8001/api/upload_and_search/', files=files)
    
    if response.status_code == 200:
        data = response.json()
        
        print("=== 유사도 검색 API 응답 ===")
        print(f"성공: {data.get('success')}")
        print(f"모드: {data.get('mode')}")
        print(f"결과 개수: {len(data.get('results', []))}")
        print()
        
        # 메타데이터 출력
        metadata = data.get('search_metadata')
        if metadata:
            print("=== 검색 메타데이터 ===")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            print()
        
        if data.get('results'):
            print("=== 첫 번째 검색 결과 ===")
            first_result = data['results'][0]
            
            for key in sorted(first_result.keys()):
                print(f"{key}: {first_result[key]}")
                
            print("\n=== 입양상태 관련 정보 ===")
            print(f"adoption_status: {first_result.get('adoption_status')}")
            print(f"adoption_status_code: {first_result.get('adoption_status_code')}")
        
    else:
        print(f"API 호출 실패: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"오류: {e}")
