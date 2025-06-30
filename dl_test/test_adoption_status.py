#!/usr/bin/env python3
import requests
import json

# API에서 강아지 데이터 가져오기
try:
    response = requests.get('http://localhost:8001/api/dogs/')
    data = response.json()
    
    if data.get('status') == 'success' and data.get('dogs'):
        first_dog = data['dogs'][0]
        
        print("=== 첫 번째 강아지 정보 ===")
        print(f"ID: {first_dog.get('id')}")
        print(f"이름: {first_dog.get('name')}")
        print(f"견종: {first_dog.get('breed')}")
        print(f"견종코드: {first_dog.get('breed_code')}")
        print(f"성별: {first_dog.get('gender')}")
        print(f"성별코드: {first_dog.get('gender_code')}")
        print(f"입양상태: {first_dog.get('adoption_status')}")
        print(f"입양상태코드: {first_dog.get('adoption_status_code')}")
        print()
        
        # 모든 키 확인
        print("=== 모든 키 ===")
        for key in sorted(first_dog.keys()):
            print(f"{key}: {first_dog[key]}")
    else:
        print("데이터 조회 실패")
        
except Exception as e:
    print(f"오류: {e}")
