import torch
import numpy as np
import os
from PIL import Image
import cv2 # OpenCV는 이미지 처리 및 시각화에 유용합니다.
import mmcv

# --- MMPose 1.3.2+ 버전에 맞는 임포트 ---
from mmpose.apis import init_model 
from mmpose.visualization import PoseLocalVisualizer 
# from mmpose.datasets import DatasetInfo # <-- 이 줄은 삭제합니다!
from mmengine import Config

# --- MMPose 1.3.2+ 버전용 구조체 임포트 ---
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData # bbox를 담기 위해 필요

# --- SimCLR 특징 검색 함수는 필요시 동적으로 임포트 ---
# 순환 참조 방지를 위해 주석 처리
# search_similar_dogs 함수는 backend에서 직접 임포트하여 사용 

# --- AP-10K 모델 및 설정 경로 ---
MMPose_ROOT = 'C:/dl_final/dl_fianl/mm_pose/mmpose' # 경로 수정: dl_fianl로 변경

# 설정 파일 경로 
AP10K_CONFIG_FILE = os.path.join(MMPose_ROOT, 'configs', 'animal_2d_keypoint', 
                                 'topdown_heatmap', 'ap10k', 'td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py') 

# 체크포인트 파일 경로 
AP10K_CHECKPOINT_FILE = os.path.join(MMPose_ROOT, 'checkpoints', 'hrnet_w32_ap10k_256x256-18aac840_20211029.pth') 

# --- SimCLR 관련 설정 (이전 파일들에서 가져옴) ---
SIMCLR_MODEL_PATH = 'models/simclr_vit_dog_model.pth' 
SIMCLR_OUT_DIM = 128
SIMCLR_IMAGE_SIZE = 224
DB_FEATURES_FILE = 'db_features.npy' 
DB_IMAGE_PATHS_FILE = 'db_image_paths.npy'

# 쿼리 이미지 경로 (테스트를 위해 설정, 실제 프로젝트에서는 사용자 입력 받음)
# 🚨🚨🚨 이 경로도 본인의 실제 이미지 경로로 변경해야 합니다! 🚨🚨🚨
# QUERY_IMAGE_PATH = 'C:\dl_final\dl_test\training\Images\n02085936-Maltese_dog\n02085936_10719.jpg'
QUERY_IMAGE_PATH = 'training/Images/n02085936-Maltese_dog/n02085936_10719.jpg'

def setup_ap10k_model(config_file=AP10K_CONFIG_FILE, checkpoint_file=AP10K_CHECKPOINT_FILE):
    """
    AP-10K 키포인트 검출 모델 및 시각화 도구를 설정합니다.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"AP-10K 설정 파일을 찾을 수 없습니다: {config_file}\n"
                                 "MMPose 설치 경로와 AP10K_CONFIG_FILE 경로를 확인하세요.")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"AP-10K 체크포인트 파일을 찾을 수 없습니다: {checkpoint_file}\n"
                                 "다운로드한 모델 파일을 AP10K_CHECKPOINT_FILE 경로에 두었는지 확인하세요.")

    device = torch.device("cpu") 
    print(f"키포인트 검출에 사용할 장치: {device}")

    cfg = Config.fromfile(config_file)
    if cfg.model.get('test_cfg') is None:
        cfg.model.test_cfg = Config(dict(flip_test=False, post_process='default', shift_heatmap=True,
                                        modulate_whr_weight=0, target_type='heatmap'))

    model = init_model(cfg, checkpoint_file, device=device)
    
    # --- PoseLocalVisualizer 초기화 ---
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(model.dataset_meta) # 모델의 메타데이터를 시각화 도구에 설정

    print("AP-10K 키포인트 검출 모델 및 시각화 도구 준비 완료.")
    # dataset_info는 더 이상 반환하지 않습니다.
    return model, device, visualizer # 반환 값 변경

def detect_and_visualize_keypoints(
    image_path: str, 
    ap10k_model, 
    device, 
    visualizer, 
    output_dir: str = 'output_keypoints'
):
    """
    단일 이미지에서 키포인트를 검출하고 시각화하여 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 원본 이미지를 RGB로 읽기 (웹 표시용)
    img_rgb = mmcv.imread(image_path, channel_order='rgb')
    if img_rgb is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {image_path}. 파일이 존재하는지 확인하세요.")
        return None, None

    img_height, img_width, _ = img_rgb.shape

    # MMPose의 고수준 추론 API 사용
    from mmpose.apis import inference_topdown
    
    # 전체 이미지에 대해 bounding box 생성 (동물 전체를 포함)
    bbox = np.array([[0, 0, img_width, img_height]], dtype=np.float32)
    
    # inference_topdown을 사용하여 키포인트 검출
    pose_results = inference_topdown(ap10k_model, image_path, bbox)
    
    if not pose_results:
        print(f"경고: {image_path}에서 키포인트를 검출할 수 없었습니다. (동물이 없거나 너무 작을 수 있음)")
        return None, None

    # 키포인트 시각화 (RGB 이미지 사용)
    vis_img_rgb = draw_advanced_keypoints_rgb(img_rgb.copy(), pose_results)
    
    base_name = os.path.basename(image_path)
    name_parts = os.path.splitext(base_name)
    output_filename = f"{name_parts[0]}_keypoints{name_parts[1]}"
    output_path = os.path.join(output_dir, output_filename)
    
    # PIL을 사용해서 RGB 이미지를 올바르게 저장
    from PIL import Image
    pil_img = Image.fromarray(vis_img_rgb)
    pil_img.save(output_path)
    print(f"키포인트 시각화 저장: {output_path}")

    # 결과를 이전 형식으로 변환
    reconstructed_pose_results = []
    for pose_result in pose_results:
        if hasattr(pose_result, 'pred_instances'):
            keypoints_xy = pose_result.pred_instances.keypoints
            keypoint_scores = pose_result.pred_instances.keypoint_scores
            
            # 텐서인 경우 numpy로 변환
            if hasattr(keypoints_xy, 'cpu'):
                keypoints_xy = keypoints_xy.cpu().numpy()
            if hasattr(keypoint_scores, 'cpu'):
                keypoint_scores = keypoint_scores.cpu().numpy()
            
            for i in range(keypoints_xy.shape[0]): 
                kpt_xy = keypoints_xy[i] 
                kpt_score = keypoint_scores[i] 
                combined_kpts = np.hstack((kpt_xy, kpt_score[:, np.newaxis]))
                reconstructed_pose_results.append({'keypoints': combined_kpts})
        else:
            # 이전 형식 그대로 사용
            reconstructed_pose_results.append(pose_result)

    return output_path, reconstructed_pose_results 

def calculate_keypoint_similarity(pose_results1, pose_results2, image_size=SIMCLR_IMAGE_SIZE):
    """
    두 키포인트 결과 간의 유사도 점수를 계산합니다.
    (간단한 예시: 키포인트 위치의 L2 거리 기반)
    """
    if not pose_results1 or not pose_results2:
        return 0.0 

    kpts1 = pose_results1[0]['keypoints'] 
    kpts2 = pose_results2[0]['keypoints']

    min_kpts = min(len(kpts1), len(kpts2)) 
    kpts1_xy = kpts1[:min_kpts, :2] 
    kpts2_xy = kpts2[:min_kpts, :2] 
    
    kpts1_xy_t = torch.from_numpy(kpts1_xy).float()
    kpts2_xy_t = torch.from_numpy(kpts2_xy).float()

    distance = torch.mean(torch.norm(kpts1_xy_t - kpts2_xy_t, dim=1)) / image_size

    similarity = float(max(0.0, 1.0 - distance.item())) 
    return similarity

def draw_advanced_keypoints_rgb(img_rgb, pose_results):
    """
    RGB 이미지에 키포인트와 골격을 그리는 함수 (투명도 30% 적용)
    """
    if not pose_results:
        return img_rgb
    
    # AP-10K 17개 키포인트 골격 연결 정의
    skeleton_connections = [
        # 머리
        (1, 2),   # 좌귀 - 우귀
        (1, 3),   # 좌귀 - 코
        (2, 4),   # 우귀 - 코  
        (0, 1),   # 코끝 - 좌귀
        (0, 2),   # 코끝 - 우귀
        
        # 몸통
        (5, 1),   # 목 - 좌귀
        (5, 2),   # 목 - 우귀
        (5, 6),   # 목 - 좌어깨
        (5, 7),   # 목 - 우어깨
        (6, 12),  # 좌어깨 - 좌엉덩이
        (7, 13),  # 우어깨 - 우엉덩이
        (12, 13), # 좌엉덩이 - 우엉덩이
        
        # 앞다리
        (6, 8),   # 좌어깨 - 좌팔꿈치
        (8, 10),  # 좌팔꿈치 - 좌발목
        (7, 9),   # 우어깨 - 우팔꿈치
        (9, 11),  # 우팔꿈치 - 우발목
        
        # 뒷다리
        (12, 14), # 좌엉덩이 - 좌무릎
        (14, 16), # 좌무릎 - 꼬리끝 (임시)
        (13, 15), # 우엉덩이 - 우무릎
        (15, 16), # 우무릎 - 꼬리끝 (임시)
    ]
    
    # BGR 색상 정의 (OpenCV는 BGR 순서)
    colors_bgr = {
        'keypoint': (0, 255, 255),      # 노란색 키포인트
        'head': (0, 0, 255),            # 🔴 빨간색 - 머리 부분
        'body': (0, 255, 0),            # 🟢 초록색 - 몸통
        'front_legs': (0, 255, 255),    # 🟡 노란색 - 앞다리
        'back_legs': (0, 165, 255),     # 🟠 주황색 - 뒷다리
    }
    
    # RGB를 BGR로 변환 (OpenCV 처리용)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 투명도 적용을 위한 오버레이 생성
    overlay = img_bgr.copy()
    alpha = 0.5  # 50% 투명도
    
    for pose_result in pose_results:
        if hasattr(pose_result, 'pred_instances'):
            keypoints = pose_result.pred_instances.keypoints
            scores = pose_result.pred_instances.keypoint_scores
            
            # 텐서인 경우 numpy로 변환
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
        else:
            keypoints = pose_result['keypoints']
            scores = pose_result.get('keypoint_scores', keypoints[..., 2:3] if keypoints.shape[-1] > 2 else None)
        
        # 첫 번째 인스턴스의 키포인트만 시각화
        if len(keypoints) > 0:
            kpts = keypoints[0]  # (17, 2) 또는 (17, 3)
            kpt_scores = scores[0] if scores is not None else np.ones(len(kpts))
            
            confidence_threshold = 0.3
            
            # 골격 선 그리기 (오버레이에)
            for connection in skeleton_connections:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(kpts) and pt2_idx < len(kpts) and 
                    kpt_scores[pt1_idx] > confidence_threshold and 
                    kpt_scores[pt2_idx] > confidence_threshold):
                    
                    pt1 = (int(kpts[pt1_idx][0]), int(kpts[pt1_idx][1]))
                    pt2 = (int(kpts[pt2_idx][0]), int(kpts[pt2_idx][1]))
                    
                    # 연결선 색상 결정
                    if connection in [(1, 2), (1, 3), (2, 4), (0, 1), (0, 2), (5, 1), (5, 2)]:
                        line_color = colors_bgr['head']
                    elif connection in [(5, 6), (5, 7), (6, 12), (7, 13), (12, 13)]:
                        line_color = colors_bgr['body']
                    elif connection in [(6, 8), (8, 10), (7, 9), (9, 11)]:
                        line_color = colors_bgr['front_legs']
                    else:
                        line_color = colors_bgr['back_legs']
                    
                    cv2.line(overlay, pt1, pt2, line_color, 3)
            
            # 키포인트 점 그리기 (오버레이에)
            for i, (kpt, score) in enumerate(zip(kpts, kpt_scores)):
                if score > confidence_threshold:
                    pt = (int(kpt[0]), int(kpt[1]))
                    
                    # 키포인트별 색상 (BGR 형식)
                    if i in [0, 1, 2, 3, 4]:  # 머리 부분
                        kpt_color = (0, 0, 255)      # 빨간색
                    elif i == 5:  # 목
                        kpt_color = (0, 255, 0)      # 초록색
                    elif i in [6, 7, 8, 9, 10, 11]:  # 앞다리
                        kpt_color = (0, 255, 255)    # 노란색
                    else:  # 뒷다리, 꼬리
                        kpt_color = (0, 165, 255)    # 주황색
                    
                    # 키포인트 원 그리기
                    cv2.circle(overlay, pt, 6, kpt_color, -1)
                    cv2.circle(overlay, pt, 6, (255, 255, 255), 2)  # 흰색 테두리
    
    # 투명도 적용하여 합성
    result = cv2.addWeighted(img_bgr, 1-alpha, overlay, alpha, 0)
    
    # BGR을 다시 RGB로 변환
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    # --- AP-10K 모델 및 시각화 도구 로드 ---
    # setup_ap10k_model의 반환 값에 dataset_info가 없습니다.
    ap10k_model, ap10k_device, visualizer = setup_ap10k_model()

    # --- SimCLR 유사 강아지 검색 ---
    query_image_path = QUERY_IMAGE_PATH 

    if not os.path.exists(query_image_path):
        print(f"오류: 쿼리 이미지 '{query_image_path}'를 찾을 수 없습니다. 경로를 수정하거나 파일을 준비해주세요.")
    else:
        print(f"\n쿼리 이미지: {query_image_path}")
        print("SimCLR 기반 유사 강아지 검색 시작...")
        
        # 동적으로 search_similar_dogs 함수 임포트 (순환 참조 방지)
        try:
            from training.search_similar_dogs import search_similar_dogs
        except ImportError:
            try:
                from .search_similar_dogs import search_similar_dogs
            except ImportError:
                from search_similar_dogs import search_similar_dogs
        
        top_similar_dogs_simclr = search_similar_dogs(
            query_image_path=query_image_path, 
            top_k=5,
            model_path=SIMCLR_MODEL_PATH,
            out_dim=SIMCLR_OUT_DIM,
            image_size=SIMCLR_IMAGE_SIZE,
            db_features_file=DB_FEATURES_FILE,
            db_image_paths_file=DB_IMAGE_PATHS_FILE
        )
        
        print("\n--- SimCLR 기반 가장 유사한 강아지 검색 결과 (상위 5개) ---")
        # 쿼리 이미지 키포인트 검출 및 시각화
        print(f"\n쿼리 이미지 키포인트 검출 및 시각화: {query_image_path}")
        # detect_and_visualize_keypoints 호출 시 dataset_info 인자 제거
        query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
            query_image_path, ap10k_model, ap10k_device, visualizer
        )

        final_results = []

        for i, (simclr_score, db_img_path) in enumerate(top_similar_dogs_simclr):
            print(f"\n--- 유사 강아지 {i+1}: {db_img_path} (SimCLR 유사도: {simclr_score:.4f}) ---")
            
            # DB 이미지 키포인트 검출 및 시각화
            # detect_and_visualize_keypoints 호출 시 dataset_info 인자 제거
            db_kp_output_path, db_pose_results = detect_and_visualize_keypoints(
                db_img_path, ap10k_model, ap10k_device, visualizer
            )

            keypoint_similarity = 0.0
            if query_pose_results and db_pose_results:
                keypoint_similarity = calculate_keypoint_similarity(query_pose_results, db_pose_results, SIMCLR_IMAGE_SIZE)
                print(f"키포인트 유사도: {keypoint_similarity:.4f}")
            else:
                print("키포인트 검출 실패 또는 결과 없음.")
            
            final_score = (simclr_score * 0.7) + (keypoint_similarity * 0.3) 
            final_results.append((final_score, db_img_path, simclr_score, keypoint_similarity))
            
            print(f"최종 복합 유사도: {final_score:.4f}")

        final_results.sort(key=lambda x: x[0], reverse=True)

        print("\n--- 최종 복합 유사도 기준 검색 결과 ---")
        for i, (final_score, path, simclr_s, kp_s) in enumerate(final_results):
            print(f"{i+1}. 최종 유사도: {final_score:.4f}, SimCLR: {simclr_s:.4f}, KP: {kp_s:.4f}, 이미지 경로: {path}")