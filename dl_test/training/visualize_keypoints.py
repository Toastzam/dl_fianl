import re
def normalize_filename(filename):
    # 파일명에서 연속된 특수문자, 공백, 괄호 등은 모두 단일 언더스코어로 치환
    name, ext = os.path.splitext(filename)
    # 1. 괄호, 공백, 특수문자 모두 _로 치환
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    # 2. 앞뒤 _ 제거
    name = name.strip('_')
    # 3. 연속된 _는 하나로 축소
    name = re.sub(r'_+', '_', name)
    return name + ext
import torch
import numpy as np
import os
from PIL import Image
import cv2 # OpenCV는 이미지 처리 및 시각화에 유용합니다.
import mmcv
import io
import requests

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

# 강아지 keypoints 기반 bounding box 추출 util 함수 (auto_pet_register.py에서 import용)
def get_dog_bbox_from_keypoints(image_path, ap10k_model, device, min_score=0.3):
    """
    이미지에서 강아지 keypoints를 검출하여 bounding box(x1, y1, x2, y2) 반환.
    keypoints 신뢰도 min_score 이상만 사용. 실패 시 None 반환.
    """
    from PIL import Image
    import numpy as np
    import cv2
    import io, requests, os
    def pad_to_square(img, fill_color=(255,255,255)):
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        new_img = Image.new('RGB', (size, size), fill_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    if image_path.startswith('http://') or image_path.startswith('https://'):
        try:
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            img_source = Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"[bbox] URL 이미지 로드 실패: {image_path}\n{e}")
            return None
    else:
        if not os.path.exists(image_path):
            print(f"[bbox] 파일 없음: {image_path}")
            return None
        img_source = Image.open(image_path).convert('RGB')

    img_source = pad_to_square(img_source)
    img_resized = img_source.resize((256, 256), Image.BILINEAR)
    img_rgb = np.array(img_resized)
    img_height, img_width = img_rgb.shape[:2]

    from mmpose.apis import inference_topdown
    bbox = np.array([[0, 0, img_width, img_height]], dtype=np.float32)
    pose_results = inference_topdown(ap10k_model, img_rgb, bbox)
    if not pose_results:
        return None
    # keypoints 추출
    if hasattr(pose_results[0], 'pred_instances'):
        keypoints = pose_results[0].pred_instances.keypoints[0]
        scores = pose_results[0].pred_instances.keypoint_scores[0]
        kpts_xy = keypoints[scores > min_score]
    else:
        kpts = pose_results[0]['keypoints']
        if kpts.shape[1] == 3:
            scores = kpts[:, 2]
            kpts_xy = kpts[scores > min_score, :2]
        else:
            kpts_xy = kpts[:, :2]
    if len(kpts_xy) == 0:
        return None
    x_min, y_min = kpts_xy.min(axis=0)
    x_max, y_max = kpts_xy.max(axis=0)
    # 여유 padding (10%)
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    x1 = max(int(x_min) - pad_x, 0)
    y1 = max(int(y_min) - pad_y, 0)
    x2 = min(int(x_max) + pad_x, img_width)
    y2 = min(int(y_max) + pad_y, img_height)
    return (x1, y1, x2, y2)
SIMCLR_MODEL_PATH = 'models/simclr_vit_dog_model_finetuned_v1.pth'
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
    output_dir: str = None,
    output_path: str = None
):
    """
    단일 이미지에서 키포인트를 검출하고 시각화하여 저장합니다.
    image_path가 http/https로 시작하면 URL에서 이미지를 불러옵니다.
    """
    # 모든 output_keypoints 저장/서빙 경로를 C:/dl_final/dl_fianl/output_keypoints로 강제 통일
    fixed_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output_keypoints'))
    print(f"[DEBUG] __file__ 위치: {os.path.abspath(__file__)}")
    print(f"[DEBUG] fixed_output_dir (dl_fianl/output_keypoints): {fixed_output_dir}")
    if not os.path.exists(fixed_output_dir):
        os.makedirs(fixed_output_dir)
    output_dir = fixed_output_dir
    # FastAPI StaticFiles 마운트 경로도 반드시 동일하게 맞춰야 함
    print(f"[INFO] 모든 output_keypoints 저장/서빙 경로를 C:/dl_final/dl_fianl/output_keypoints로 강제 통일합니다.")

    # --- 이미지 로딩 (로컬/URL 모두 지원) 및 256x256 리사이즈 ---
    img_rgb = None
    img_source = None
    def pad_to_square(img, fill_color=(255,255,255)):
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        new_img = Image.new('RGB', (size, size), fill_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    if image_path.startswith('http://') or image_path.startswith('https://'):
        try:
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            img_source = Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"오류: URL에서 이미지를 불러올 수 없습니다: {image_path}\n{e}")
            return None, None
    else:
        if not os.path.exists(image_path):
            print(f"오류: 이미지를 로드할 수 없습니다: {image_path}. 파일이 존재하는지 확인하세요.")
            return None, None
        img_source = Image.open(image_path).convert('RGB')

    # 정사각형 패딩 후 256x256 리사이즈 및 디버그 이미지 저장
    img_source = pad_to_square(img_source)
    # 디버그: 패딩된 이미지 저장
    debug_pad_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_padded.jpg")
    img_source.save(debug_pad_path)
    print(f"[DEBUG] 패딩 이미지 저장: {debug_pad_path}")

    img_resized = img_source.resize((256, 256), Image.BILINEAR)
    # 디버그: 리사이즈된 이미지 저장
    debug_resize_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_resized256.jpg")
    Image.fromarray(np.array(img_resized)).save(debug_resize_path)
    print(f"[DEBUG] 256x256 리사이즈 이미지 저장: {debug_resize_path}")

    img_rgb = np.array(img_resized)
    print(f"[DEBUG] img_rgb shape: {img_rgb.shape}, dtype: {img_rgb.dtype}, min: {img_rgb.min()}, max: {img_rgb.max()}")

    # --- 디버깅: 전처리된 이미지 shape, dtype, min/max 값 출력 및 저장 ---
    print(f"[DEBUG] Preprocessed image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}, min: {img_rgb.min()}, max: {img_rgb.max()}")
    # 중간 이미지 저장 (디버깅용)
    debug_padded_path = os.path.join(output_dir, 'debug_padded_' + os.path.basename(image_path))
    debug_resized_path = os.path.join(output_dir, 'debug_resized_' + os.path.basename(image_path))
    try:
        img_source.save(debug_padded_path)
        Image.fromarray(img_rgb).save(debug_resized_path)
        print(f"[DEBUG] Saved padded image: {debug_padded_path}")
        print(f"[DEBUG] Saved resized image: {debug_resized_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to save debug images: {e}")
    img_height, img_width = img_rgb.shape[:2]
    # 원본 크기 정보도 pose_results에 포함 (아래에서 img_width, img_height로 저장)

    # MMPose의 고수준 추론 API 사용
    from mmpose.apis import inference_topdown
    bbox = np.array([[0, 0, img_width, img_height]], dtype=np.float32)

    # inference_topdown은 numpy array도 입력 가능

    pose_results = inference_topdown(ap10k_model, img_rgb, bbox)

    # --- 디버깅: pose_results가 비었을 때 shape, dtype, 파일 존재 여부 출력 ---
    if not pose_results:
        print(f"[DEBUG] pose_results is empty for {image_path}")
        print(f"[DEBUG] Preprocessed image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}, min: {img_rgb.min()}, max: {img_rgb.max()}")
        print(f"[DEBUG] Does padded image exist? {os.path.exists(debug_padded_path)}")
        print(f"[DEBUG] Does resized image exist? {os.path.exists(debug_resized_path)}")
        print(f"경고: {image_path}에서 키포인트를 검출할 수 없었습니다. (동물이 없거나 너무 작을 수 있음)")
        return None, None

    vis_img_rgb = draw_advanced_keypoints_rgb(img_rgb.copy(), pose_results)

    # 파일명 생성 (URL이면 파일명만 추출, 확장자는 항상 .jpg로 강제)
    if output_path is not None:
        # 외부에서 경로를 지정한 경우 해당 경로에 저장 (정규화 적용)
        output_path = os.path.abspath(output_path)
        output_path = os.path.join(os.path.dirname(output_path), normalize_filename(os.path.basename(output_path)))
    else:
        base_name = os.path.basename(image_path)
        if not base_name:
            base_name = 'remote_image'
        base_name = normalize_filename(base_name)
        name_parts = os.path.splitext(base_name)
        output_filename = f"{name_parts[0]}_keypoints.jpg"  # 항상 jpg로 저장
        output_path = os.path.join(output_dir, output_filename)

    pil_img = Image.fromarray(vis_img_rgb)
    pil_img.save(output_path, format='JPEG')
    print(f"키포인트 시각화 저장: {output_path}")

    # 결과를 이전 형식으로 변환 (width, height 정보 포함)
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
                reconstructed_pose_results.append({'keypoints': combined_kpts, 'img_width': img_width, 'img_height': img_height})
        else:
            # 이전 형식 그대로 사용, width/height 추가
            pose_dict = dict(pose_result)
            pose_dict['img_width'] = img_width
            pose_dict['img_height'] = img_height
            reconstructed_pose_results.append(pose_dict)

    # pose_results가 비어있으면 None 반환 (언팩 에러 방지)
    if not reconstructed_pose_results:
        return None, None
    return output_path, reconstructed_pose_results

def calculate_keypoint_similarity(pose_results1, pose_results2, image_size=SIMCLR_IMAGE_SIZE):
    """
    두 키포인트 결과 간의 유사도 점수를 계산합니다.
    (간단한 예시: 키포인트 위치의 L2 거리 기반)
    """
    if not pose_results1 or not pose_results2:
        print("[DEBUG] calculate_keypoint_similarity: pose_results1 or pose_results2 is empty")
        return 0.0

    kpts1 = pose_results1[0]['keypoints']
    kpts2 = pose_results2[0]['keypoints']

    min_kpts = min(len(kpts1), len(kpts2))
    kpts1_xy = kpts1[:min_kpts, :2]
    kpts2_xy = kpts2[:min_kpts, :2]

    # 각 이미지의 width, height로 정규화
    w1 = pose_results1[0].get('img_width', image_size)
    h1 = pose_results1[0].get('img_height', image_size)
    w2 = pose_results2[0].get('img_width', image_size)
    h2 = pose_results2[0].get('img_height', image_size)

    # (x, y) 각각 정규화
    kpts1_xy_norm = np.zeros_like(kpts1_xy)
    kpts2_xy_norm = np.zeros_like(kpts2_xy)
    kpts1_xy_norm[:, 0] = kpts1_xy[:, 0] / w1
    kpts1_xy_norm[:, 1] = kpts1_xy[:, 1] / h1
    kpts2_xy_norm[:, 0] = kpts2_xy[:, 0] / w2
    kpts2_xy_norm[:, 1] = kpts2_xy[:, 1] / h2

    # print(f"[DEBUG] kpts1_xy_norm: {kpts1_xy_norm}")
    # print(f"[DEBUG] kpts2_xy_norm: {kpts2_xy_norm}")  # [자동 주석처리] kpts2_xy_norm 디버그 출력

    # NaN/Inf 체크
    if np.any(np.isnan(kpts1_xy_norm)) or np.any(np.isnan(kpts2_xy_norm)):
        print("[DEBUG] NaN detected in normalized keypoints!")
        return 0.0
    if np.any(np.isinf(kpts1_xy_norm)) or np.any(np.isinf(kpts2_xy_norm)):
        print("[DEBUG] Inf detected in normalized keypoints!")
        return 0.0

    kpts1_xy_t = torch.from_numpy(kpts1_xy_norm).float()
    kpts2_xy_t = torch.from_numpy(kpts2_xy_norm).float()

    distance = torch.mean(torch.norm(kpts1_xy_t - kpts2_xy_t, dim=1))
    print(f"[DEBUG] keypoint L2 distance: {distance.item()}")

    # distance가 0에 가까울수록 유사도 1, distance가 커질수록 유사도 0에 가까워짐
    similarity = float(1.0 / (1.0 + distance.item()))
    print(f"[DEBUG] keypoint similarity (1/(1+d)): {similarity}")
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
                    
                    cv2.line(overlay, pt1, pt2, line_color, 1, lineType=cv2.LINE_AA)
            
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
                    
                    # 키포인트 원 그리기 (크기 축소)
                    cv2.circle(overlay, pt, 3, kpt_color, -1)
                    cv2.circle(overlay, pt, 3, (255, 255, 255), 1)  # 흰색 테두리
    
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
        query_kp_output_path, query_pose_results = detect_and_visualize_keypoints(
            query_image_path, ap10k_model, ap10k_device, visualizer
        )
        print("[DEBUG] 쿼리 pose_results type:", type(query_pose_results))
        print("[DEBUG] 쿼리 pose_results len:", len(query_pose_results) if query_pose_results else 0)
        if query_pose_results:
            print("[DEBUG] 쿼리 pose_results[0] type:", type(query_pose_results[0]))
            print("[DEBUG] 쿼리 pose_results[0] keys:", list(query_pose_results[0].keys()) if isinstance(query_pose_results[0], dict) else "(not dict)")
            if isinstance(query_pose_results[0], dict) and 'keypoints' in query_pose_results[0]:
                print("[DEBUG] 쿼리 keypoints shape:", query_pose_results[0]['keypoints'].shape)
                print("[DEBUG] 쿼리 keypoints sample:\n", query_pose_results[0]['keypoints'])
            else:
                print("[DEBUG] 쿼리 pose_results[0]에 'keypoints' 없음 또는 타입 불일치")
        else:
            print("[DEBUG] 쿼리 pose_results가 None 또는 빈 리스트임")

        final_results = []

        # SimCLR 검색 결과가 dict 또는 튜플일 수 있으므로 자동 판별
        for i, result in enumerate(top_similar_dogs_simclr):
            if isinstance(result, dict):
                simclr_score = result.get('similarity', 0.0)
                db_img_path = result.get('image_url') or result.get('image_path')
            else:
                simclr_score, db_img_path = result

            print(f"\n--- 유사 강아지 {i+1}: {db_img_path} (SimCLR 유사도: {simclr_score:.4f}) ---")
            db_kp_output_path, db_pose_results = detect_and_visualize_keypoints(
                db_img_path, ap10k_model, ap10k_device, visualizer
            )

            print(f"[DEBUG] DB pose_results for {db_img_path} type:", type(db_pose_results))
            print(f"[DEBUG] DB pose_results for {db_img_path} len:", len(db_pose_results) if db_pose_results else 0)
            if db_pose_results:
                print(f"[DEBUG] DB pose_results[0] type:", type(db_pose_results[0]))
                print(f"[DEBUG] DB pose_results[0] keys:", list(db_pose_results[0].keys()) if isinstance(db_pose_results[0], dict) else "(not dict)")
                if isinstance(db_pose_results[0], dict) and 'keypoints' in db_pose_results[0]:
                    print(f"[DEBUG] DB keypoints shape:", db_pose_results[0]['keypoints'].shape)
                    print(f"[DEBUG] DB keypoints sample:\n", db_pose_results[0]['keypoints'])
                else:
                    print(f"[DEBUG] DB pose_results[0]에 'keypoints' 없음 또는 타입 불일치")
            else:
                print(f"[DEBUG] DB pose_results가 None 또는 빈 리스트임")

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