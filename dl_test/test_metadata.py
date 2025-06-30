#!/usr/bin/env python3
import requests
import json
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
        
        print("=== 전체 API 응답 ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
    else:
        print(f"API 호출 실패: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"오류: {e}")
