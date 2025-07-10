import os
import json
import datetime
from database import get_all_pet_images

LAST_TRAIN_STATE_FILE = os.path.join(os.path.dirname(__file__), '..', 'last_training_state.json')
RETRAIN_TRIGGER_COUNT = 50  # 50개의 새 이미지가 추가되면 재학습 트리거 (필요시 조정)

# 마지막 학습 상태를 불러옴
def load_last_training_state():
    if os.path.exists(LAST_TRAIN_STATE_FILE):
        with open(LAST_TRAIN_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_image_count": 0, "last_image_ids": [], "last_trained_image_id": None}

# 마지막 학습 상태를 저장
def save_last_training_state(state):
    with open(LAST_TRAIN_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# 마지막 학습 이후 새로 추가된 이미지를 반환
def get_new_images_since_last_training():
    pet_images = get_all_pet_images()
    last_state = load_last_training_state()
    last_ids = set(last_state.get("last_image_ids", []))
    # 각 이미지 딕셔너리에 고유 'id'가 있으면 사용, 없으면 'public_url' 또는 'image_path' 사용
    new_images = [img for img in pet_images if get_image_unique_id(img) not in last_ids]
    return new_images

# 이미지의 고유 ID 반환 (DB id > public_url > image_path)
def get_image_unique_id(img):
    return img.get('id') or img.get('public_url') or img.get('image_path')

# 재학습 트리거 조건 충족 여부 반환
def should_trigger_retraining():
    new_images = get_new_images_since_last_training()
    return len(new_images) >= RETRAIN_TRIGGER_COUNT

# 필요시 재학습 트리거 (특징 추출 후 호출)
def trigger_retraining_if_needed():
    """
    새로운 이미지 특징 추출 후 호출. 새 이미지가 충분히 쌓이면 재학습 및 상태 갱신.
    오후 11시(23시) 이후에만 재학습이 실행됩니다.
    """
    now = datetime.datetime.now()
    if now.hour < 23:
        print(f"[RetrainManager] 현재 시간({now.hour}시)에는 재학습이 실행되지 않습니다. (23시 이후에만 허용)")
        return
    if should_trigger_retraining():
        print("[RetrainManager] 새 이미지가 충분히 쌓여 재학습을 시작합니다...")
        new_images = get_new_images_since_last_training()
        # 실제 재학습 로직 호출 (예: 서브프로세스 실행, 학습 스크립트 임포트 등)
        retrain_model_with_new_images(new_images)
        # 마지막 학습 상태 갱신
        pet_images = get_all_pet_images()
        all_ids = [get_image_unique_id(img) for img in pet_images]
        state = {
            "last_image_count": len(pet_images),
            "last_image_ids": all_ids,
            "last_trained_image_id": all_ids[-1] if all_ids else None
        }
        save_last_training_state(state)
        print("[RetrainManager] 재학습 완료 및 상태 갱신.")
    else:
        print("[RetrainManager] 새 이미지가 충분하지 않아 재학습을 건너뜁니다.")

# 실제 재학습 로직 (여기서 학습 스크립트 실행 등 구현)
def retrain_model_with_new_images(new_images):
    print(f"[RetrainManager] {len(new_images)}개의 새 이미지로 모델을 재학습합니다...")
    # 실제 SimCLR 파인튜닝 스크립트 실행 (아직 활성화 X, 아래 주석 해제 시 동작)
    # import subprocess
    # subprocess.run(['python', '../training/train_simclr_finetune.py'])
    # 또는
    # os.system('python ../training/train_simclr_finetune.py')
    print("[RetrainManager] (시뮬레이션) 모델 재학습 완료. (실제 파인튜닝은 주석처리되어 있음)")
