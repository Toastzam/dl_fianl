import os
import requests
import shutil
import pymysql  # 또는 sqlite3 등 사용 가능

# --- DB 연결 정보 ---
DB_HOST = 'byhou.synology.me'
DB_USER = 'h3'
DB_PASSWORD = 'Dbrlrus25^'
DB_NAME = 'h3'
TABLE_NAME = 'pet_image'
IMAGE_FIELD = 'public_url'  # 실제 컬럼명에 맞게 수정


# --- 저장 폴더 ---
SAVE_DIR = 'training/Images'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- DB에서 이미지 경로/URL 가져오기 ---
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8', port=3370)
cur = conn.cursor()
cur.execute(f"SELECT {IMAGE_FIELD} FROM {TABLE_NAME}")
rows = cur.fetchall()
cur.close()
conn.close()

print(f"총 {len(rows)}개 이미지 다운로드 시작...")

for idx, (img_path,) in enumerate(rows):
    try:
        # 파일명만 추출, 중복 방지 위해 idx 추가
        ext = os.path.splitext(img_path)[-1]
        save_name = f"img_{idx}{ext if ext else '.jpg'}"
        save_path = os.path.join(SAVE_DIR, save_name)

        if img_path.startswith('http://') or img_path.startswith('https://'):
            # URL에서 다운로드
            resp = requests.get(img_path, timeout=10)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(resp.content)
        else:
            # 로컬 파일 복사
            if os.path.exists(img_path):
                shutil.copy(img_path, save_path)
            else:
                print(f"[경고] 파일 없음: {img_path}")
                continue
        print(f"[{idx+1}/{len(rows)}] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[{idx+1}/{len(rows)}] 실패: {img_path} ({e})")

print("모든 이미지 다운로드/복사 완료!")