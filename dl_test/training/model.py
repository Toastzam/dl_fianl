import torch
import torch.nn as nn
import timm # PyTorch Image Models 라이브러리
from transformers import ViTModel


class SimCLRVIT(nn.Module):
    def __init__(self, out_dim):
        super(SimCLRVIT, self).__init__()
        # ViT 백본 모델을 로드합니다.
        # 여기서는 더 작은 ViT-Small 모델을 사용합니다.
        # 'vit_base_patch16_224' 대신 'vit_small_patch16_224' 또는 'vit_tiny_patch16_224' 사용
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        
         # 백본의 출력 차원에 맞게 첫 번째 레이어의 입력 차원을 설정해야 합니다.
        # vit_small_patch16_224의 출력 차원은 768입니다.
        projection_head_input_dim = self.backbone.num_features # 또는 768로 직접 지정

        # 2. SimCLR Projection Head 정의
        # SimCLR 논문에 따라 MLP (Multi-Layer Perceptron) 구조를 사용합니다.
        # feature_dim -> feature_dim -> out_dim (선형 -> ReLU -> 선형)
        self.projection_head = nn.Sequential(
            nn.Linear(projection_head_input_dim, projection_head_input_dim),
            nn.ReLU(),
            nn.Linear(projection_head_input_dim, out_dim) # out_dim은 128로 설정되어 있습니다.
        )

    def forward(self, x):
        features = self.backbone(x)
        # ViT는 클래스 토큰만 반환하거나, 풀링된 피처를 반환할 수 있습니다.
        # timm의 num_classes=0 설정 시, 기본적으로 마지막 토큰(클래스 토큰)의 피처를 반환합니다.
        # 만약 전역 평균 풀링(GAP)을 원한다면, 모델 구조에 따라 추가적인 처리가 필요할 수 있습니다.
        # 현재는 num_classes=0이면 풀링된 피처 또는 마지막 레이어 이전 피처를 반환합니다.
        
        # 프로젝션 헤드를 통과시켜 SimCLR 임베딩을 얻습니다.
        projection = self.projection_head(features)
        return projection

# --- 모델 테스트 (이 부분은 train.py에 포함될 내용입니다) ---
if __name__ == "__main__":
    # 모델 인스턴스 생성 (기본 ViT Base, 출력 차원 128)
    model = SimCLRVIT(out_dim=128)
    
    # GPU가 사용 가능하다면 모델을 GPU로 보냅니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 가상의 입력 데이터 생성 (예: 배치 크기 4, 채널 3, 이미지 크기 224x224)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # 모델에 입력하여 결과 확인
    with torch.no_grad(): # 테스트 시에는 기울기 계산을 비활성화합니다.
        output_embeddings = model(dummy_input)
    
    print(f"모델 출력 임베딩 텐서 크기: {output_embeddings.shape}") # 예상: [4, 128]
    print("모델 정의 및 테스트 완료.")