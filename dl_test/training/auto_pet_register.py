
from collections import Counter
import cv2
# mmpose keypoint 기반 bounding box util import
from visualize_keypoints import setup_ap10k_model, get_dog_bbox_from_keypoints

# mmpose keypoint 기반 bounding box crop dominant color 추출
def get_dominant_color(image_path, ap10k_model=None, ap10k_device=None, k=3):
    img = cv2.imread(image_path)
    if img is None:
        return random.choice(colors)
    # bounding box 추출 (모델이 준비된 경우만)
    bbox = None
    if ap10k_model is not None and ap10k_device is not None:
        try:
            bbox = get_dog_bbox_from_keypoints(image_path, ap10k_model, ap10k_device)
        except Exception as e:
            print(f"[경고] keypoint bbox 추출 실패: {e}")
    if bbox:
        x1, y1, x2, y2 = bbox
        img = img[y1:y2, x1:x2]
        if img.size == 0:
            img = cv2.imread(image_path)  # fallback
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.reshape((-1, 3))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    counts = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_
    dominant_color = center_colors[counts.most_common(1)[0][0]]
    r, g, b = [int(x) for x in dominant_color]
    import numpy as np
    color_np = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(color_np, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    # 밝기/채도/색상 기준 개선
    if v < 50:
        return "검정색"
    if s < 30 and v > 200:
        return "흰색"
    if 30 <= s < 60 and v > 200:
        return "크림색"
    if s < 40:
        return "회색"
    # hue 기준 더 세분화
    if (h < 15 or h >= 170):
        if v > 180:
            return "크림색"
        return "붉은색"
    elif 15 <= h < 25:
        if v > 200:
            return "베이지"
        return "노란색"
    elif 25 <= h < 35:
        return "노란색"
    elif 35 <= h < 85:
        return "초록색"
    elif 85 <= h < 130:
        return "푸른색"
    elif 130 <= h < 170:
        return "보라색"
    return random.choice(colors)

import os
import random
import requests
import json
from datetime import datetime

# 프론트엔드와 맞는 필드 및 코드 매핑 필요
API_BASE = "http://192.168.0.38:8081"  # 실제 서버 주소로 변경 (루트)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
NUM_TO_REGISTER = 500 # 등록할 이미지 개수


features = [
"활발하고 명랑함", "사람을 잘 따름", "조용하고 온순함", "호기심이 많음", "장난꾸러기",
"겁이 많음", "애교가 많음", "지능이 높음", "잘 짖지 않음", "산책을 좋아함",
"충성심이 강함", "경계심이 강함", "먹성이 좋음", "활동량이 많음", "잠이 많음",
"칭찬에 약함", "고집이 셈", "분리불안 경향", "새로운 것을 좋아함", "깔끔함",
"수영을 좋아함", "훈련 습득이 빠름", "낯가림이 있음", "집 지키는 것을 잘함", "다른 동물과 잘 지냄",
"털갈이가 심함", "피부병에 취약함", "더위를 많이 탐", "추위를 많이 탐", "식탐이 많음",
"배변 훈련이 쉬움", "털 관리가 필요함", "사회성이 좋음", "독립적인 성격", "온순한 성격",
"고양이와 잘 지냄", "어린아이와 잘 지냄", "노인과 잘 지냄", "집안을 어지럽힘", "물건을 물어뜯음",
"울음 소리가 큼", "낑낑거림", "칭찬을 좋아함", "혼자 있는 것을 싫어함", "겁이 없음",
"사람을 경계함", "낯선 사람에게 공격적", "온순하고 착함", "사냥 본능이 강함", "겁이 많아 숨는 것을 좋아함",
"먹이를 가림", "식성이 까다로움", "간식에 대한 집착", "물을 무서워함", "차 타는 것을 싫어함",
"미용을 싫어함", "귀 청소를 싫어함", "양치질을 싫어함", "발 만지는 것을 싫어함", "병원 가는 것을 싫어함",
"낯선 소리에 예민함", "천둥번개에 겁먹음", "산책 시 당김", "줄을 싫어함", "목욕을 싫어함",
"자기 영역 표시를 잘함", "마킹을 많이 함", "털이 잘 빠지지 않음", "알레르기가 적음", "침을 많이 흘림",
"코를 많이 골음", "자주 헥헥거림", "자주 기침함", "변비가 있음", "설사를 자주 함",
"구토를 자주 함", "식사 속도가 빠름", "음식을 급하게 먹음", "입이 짧음", "사료를 잘 안 먹음",
"소리에 민감함", "냄새에 민감함", "시력이 좋음", "후각이 발달함", "귀가 밝음",
"꼬리 흔들기를 잘함", "애정 표현이 풍부함", "배를 보이는 것을 좋아함", "안기는 것을 좋아함", "핥는 것을 좋아함",
"응석받이", "질투심이 많음", "칭찬에 보답하려 함", "주인을 지키려 함", "낯선 환경에 잘 적응함",
"낯선 환경에 민감함", "사회화가 잘 되어 있음", "사회화가 부족함", "활동량이 적음", "산책을 귀찮아함",
"간식만 좋아함", "놀이를 좋아함", "공놀이를 좋아함", "터그놀이를 좋아함", "숨바꼭질을 좋아함"
]

colors = [
"흰색", "갈색", "검정색", "회색", "베이지", "황색", "크림색", "붉은색", "은색", "푸른색",
"검정&흰색", "갈색&흰색", "황색&흰색", "회색&흰색", "붉은색&흰색", "검정&갈색", "회색&갈색", "황색&갈색", "붉은색&갈색", "베이지&갈색",
"트라이컬러 (삼색)", "블랙탄", "초콜릿", "블루멀", "레드멀", "세이블", "브린들", "파티컬러", "멀", "데님블루"
]

dog_names = [
"코코", "초코", "보리", "두부", "사랑이", "별이", "행복이", "구름이", "몽이", "뽀미",
"루루", "바둑이", "쫑이", "순이", "하늘이", "달이", "깜이", "탄이", "라떼", "밀키",
"밤이", "솜이", "유리", "마루", "똘이", "복실이", "쏭이", "단이", "누리", "호두",
"칸", "루비", "토리", "해피", "메리", "제리", "팡이", "짱아", "뭉치", "보배",
"바비", "쿠키", "똘비", "두리", "라온", "금동이", "레오", "베리", "봉구", "테리",
"루키", "로이", "코난", "순돌이", "단추", "꾸미", "땡이", "슈슈", "포포", "찰리",
"바다", "송이", "심바", "도담", "하니", "까미", "단비", "몽이", "하루", "쪼꼬",
"모찌", "봉식", "대박", "은비", "복동이", "솔이", "초롱이", "곰돌이", "방울이", "꼬미",
"방울", "몽실이", "코이", "뿌꾸", "오드리", "아리", "나나", "체리", "진저", "샤샤",
"망고", "카푸", "치즈", "럭키", "미미", "소라", "호야", "미르", "유키", "까꿍",
"달래", "로지", "포도", "쿠마", "마리", "퐁이", "라미", "탱이", "앙꼬", "버터",
"크림", "슈가", "꼬미", "다온", "라온", "새롬", "새벽", "해랑", "솔이", "미소",
"누리", "다나", "단풍", "마음", "사랑", "소망", "믿음", "행복", "기쁨", "나비",
"삐삐", "봉순이", "덕구", "땡구", "순돌이", "철수", "영희", "희망이", "소금이", "후추",
"바닐라", "아이스", "젤리", "푸딩", "마카롱", "시루", "두리", "예삐", "뚱이", "땡칠이"
]
  
def normalize_breed_name(name):
    return name.replace("_", "").replace("-", "").replace(" ", "").lower()

def get_common_codes():
    codes = requests.get(f"{API_BASE}/api/common/codes").json()
    shelters = requests.get(f"{API_BASE}/api/common/shelters").json()
    # 견종 코드 매핑: cd_nm_en(영문명) 기준, 유연 매칭
    breed_map = {}
    for c in codes:
        if c['groupCd'] == 'DOG_BREED':
            en_name = c.get('cd_nm_en') or c.get('cdNmEn') or c.get('cd_nm') or c.get('cdNm')
            if en_name:
                breed_map[normalize_breed_name(en_name)] = c['cd']
    gender_codes = [c['cd'] for c in codes if c['groupCd'] == 'PET_GENDER']
    neuter_codes = [c['cd'] for c in codes if c['groupCd'] == 'NEUTER_STATUS']
    adoption_codes = [c['cd'] for c in codes if c['groupCd'] == 'ADOPTION_STATUS']
    orgs = list(set([s['jurisdictionOrg'] for s in shelters]))
    return breed_map, gender_codes, neuter_codes, adoption_codes, shelters, orgs

def extract_breed(folder_name):
    # 예: n02085936-Maltese_dog → Maltese
    if "-" in folder_name:
        breed = folder_name.split("-")[1]
        breed = breed.replace("_dog", "").replace("_", " ")
        return breed
    return folder_name

def main():
    breed_map, gender_codes, neuter_codes, adoption_codes, shelters, orgs = get_common_codes()
    # mmpose 모델 준비 (최초 1회)
    print("[INFO] mmpose keypoint 모델 로딩 중...")
    ap10k_model, ap10k_device, _ = setup_ap10k_model()
    print("[INFO] mmpose keypoint 모델 준비 완료.")

    image_paths = []
    for breed_folder in os.listdir(IMAGES_DIR):
        breed_path = os.path.join(IMAGES_DIR, breed_folder)
        if os.path.isdir(breed_path):
            for img in os.listdir(breed_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append((os.path.join(breed_path, img), breed_folder))
    random.shuffle(image_paths)

    success_count = 0
    tried = 0
    for img_path, breed_folder in image_paths:
        if success_count >= NUM_TO_REGISTER:
            break
        breed_name_raw = extract_breed(breed_folder)
        breed_name = normalize_breed_name(breed_name_raw)
        breedCd = breed_map.get(breed_name)
        if not breedCd:
            print(f"[경고] '{breed_name_raw}' → '{breed_name}' 품종 매핑 실패. 307(Mixed)로 대체: {img_path}")
            breedCd = "307"  # Mixed
        shelter = random.choice(shelters)
        # 2021-01 ~ 2025-01 랜덤 월 생성
        year = random.randint(2021, 2025)
        if year == 2025:
            month = 1
        else:
            month = random.randint(1, 12)
        birthYyyyMm = f"{year}-{month:02d}"
        color_name = get_dominant_color(img_path, ap10k_model, ap10k_device)
        gender_choices = ["M", "F", "Q"]  # 수컷, 암컷, 미상
        genderCd = random.choice(gender_choices)
        data = {
            "petUid": "",
            "name": random.choice(dog_names),
            "birthYyyyMm": birthYyyyMm,
            "receptionDate": datetime.now().strftime("%Y-%m-%d"),
            "weightKg": str(round(random.uniform(4, 10), 1)),
            "color": color_name,
            "genderCd": genderCd,  # 랜덤 성별
            "breedCd": breedCd,
            "neuteredCd": "U",  # 미상
            "adoptionStatusCd": "APPLY_AVAILABLE",
            "jurisdictionOrg": shelter['jurisdictionOrg'],
            "shelterId": str(shelter['shelterId']),
            "foundLocation": shelter.get('shelterRoadAddr', ''),
            "feature": random.choice(features),
            "noticeId": ""
        }
        with open(img_path, "rb") as img_file:
            files = [
                ("petProfile", (None, json.dumps(data), "application/json")),
                ("photos", (os.path.basename(img_path), img_file, "image/jpeg")),
            ]
            response = requests.post(
                f"{API_BASE}/manager/pet",
                files=files
            )
            print(f"{img_path} 등록 결과: {response.status_code}")
            if response.status_code == 200:
                success_count += 1
        tried += 1
    print(f"총 {tried}개 시도, {success_count}개 성공 등록")

if __name__ == "__main__":
    main()
