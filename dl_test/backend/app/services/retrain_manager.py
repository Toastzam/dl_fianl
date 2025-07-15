"""
재학습 관리 서비스
"""
import os
import json
import datetime
from typing import List, Dict, Any

class RetrainManagerService:
    """모델 재학습 관리 서비스"""
    
    def __init__(self):
        self.state_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'last_training_state.json'
        )
        self.trigger_count = 50  # 50개의 새 이미지가 추가되면 재학습 트리거
    
    def load_last_training_state(self) -> Dict[str, Any]:
        """마지막 학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_image_count": 0, 
            "last_image_ids": [], 
            "last_trained_image_id": None
        }
    
    def save_last_training_state(self, state: Dict[str, Any]):
        """마지막 학습 상태 저장"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def get_image_unique_id(self, img: Dict[str, Any]) -> str:
        """이미지의 고유 ID 반환"""
        return img.get('id') or img.get('public_url') or img.get('image_path')
    
    def get_new_images_since_last_training(self) -> List[Dict[str, Any]]:
        """마지막 학습 이후 새로 추가된 이미지 반환"""
        from app.database import get_all_pet_images
        
        pet_images = get_all_pet_images()
        last_state = self.load_last_training_state()
        last_ids = set(last_state.get("last_image_ids", []))
        
        new_images = [
            img for img in pet_images 
            if self.get_image_unique_id(img) not in last_ids
        ]
        return new_images
    
    def should_trigger_retraining(self) -> bool:
        """재학습 트리거 조건 충족 여부 반환"""
        new_images = self.get_new_images_since_last_training()
        return len(new_images) >= self.trigger_count
    
    def trigger_retraining_if_needed(self):
        """필요시 재학습 트리거"""
        now = datetime.datetime.now()
        if now.hour < 23:
            print(f"[RetrainManager] 현재 시간({now.hour}시)에는 재학습이 실행되지 않습니다. (23시 이후에만 허용)")
            return
        
        if self.should_trigger_retraining():
            print("[RetrainManager] 새 이미지가 충분히 쌓여 재학습을 시작합니다...")
            new_images = self.get_new_images_since_last_training()
            
            # 실제 재학습 로직 호출
            self.retrain_model_with_new_images(new_images)
            
            # 마지막 학습 상태 갱신
            from app.database import get_all_pet_images
            pet_images = get_all_pet_images()
            all_ids = [self.get_image_unique_id(img) for img in pet_images]
            
            state = {
                "last_image_count": len(pet_images),
                "last_image_ids": all_ids,
                "last_trained_image_id": all_ids[-1] if all_ids else None
            }
            self.save_last_training_state(state)
            print("[RetrainManager] 재학습 완료 및 상태 갱신.")
        else:
            print("[RetrainManager] 새 이미지가 충분하지 않아 재학습을 건너뜁니다.")
    
    def retrain_model_with_new_images(self, new_images: List[Dict[str, Any]]):
        """실제 재학습 로직"""
        print(f"[RetrainManager] {len(new_images)}개의 새 이미지로 모델을 재학습합니다...")
        
        # 실제 SimCLR 파인튜닝 스크립트 실행 (필요시 주석 해제)
        # import subprocess
        # subprocess.run(['python', '../training/train_simclr_finetune.py'])
        # 또는
        # os.system('python ../training/train_simclr_finetune.py')
        
        print("[RetrainManager] (시뮬레이션) 모델 재학습 완료. (실제 파인튜닝은 주석처리되어 있음)")

# 전역 인스턴스
_retrain_manager = None

def get_retrain_manager() -> RetrainManagerService:
    """재학습 관리자 인스턴스 반환"""
    global _retrain_manager
    if _retrain_manager is None:
        _retrain_manager = RetrainManagerService()
    return _retrain_manager

# 기존 호환성을 위한 함수들
def trigger_retraining_if_needed():
    """기존 호환성을 위한 래퍼 함수"""
    manager = get_retrain_manager()
    manager.trigger_retraining_if_needed()
