import torch
import transformers
import PIL
import numpy
import sklearn
import matplotlib
import cv2

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")

print(f"Transformers 버전: {transformers.__version__}")
print(f"Pillow 버전: {PIL.__version__}")
print(f"NumPy 버전: {numpy.__version__}")
print(f"Scikit-learn 버전: {sklearn.__version__}")
print(f"Matplotlib 버전: {matplotlib.__version__}")
print(f"OpenCV 버전: {cv2.__version__}")

print("\n--- 모든 핵심 라이브러리 임포트 성공! ---")