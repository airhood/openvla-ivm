#!/bin/bash
# research/setup_ubuntu_env.sh
#
# Ubuntu(로컬)에서 robosuite/LIBERO 데이터 생성 파이프라인을 돌리기 위한 venv 설정.
#
# 배경: Colab 컨테이너에 NVIDIA용 EGL 렌더링 드라이버(vendor ICD)가 없어서(Mesa만 있음),
# robosuite가 하드코딩해서 쓰는 EGL offscreen 렌더링이 계속 세그폴트로 죽었다. 렌더링
# 단계만 로컬 Ubuntu로 옮기기로 함 — VLA forward pass/LIVModule 학습은 계속 Colab에서.
# 자세한 경위는 docs/MODEL.md 참고.
#
# 사용법: 이 저장소 루트에서 실행
#   bash research/setup_ubuntu_env.sh
#
# 완료 후:
#   source venv/bin/activate
#   python research/data_generation/smoke_test.py --task_suite_name libero_spatial --task_id 0
#
# 참고: Ubuntu가 실제 데스크톱 세션(진짜 디스플레이 있음)이면, Colab에서 겪었던
# MUJOCO_GL/Xvfb 문제 자체가 없을 가능성이 높다 — 아무 env var 없이 먼저 시도해볼 것.
# 헤드리스(SSH만)로 쓰는 경우에만 EGL/OSMesa/Xvfb 설정이 다시 필요할 수 있음.

set -e

PYTHON_BIN=${PYTHON_BIN:-python3.10}

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "오류: $PYTHON_BIN 을 찾을 수 없음."
    echo "설치: sudo apt-get install python3.10 python3.10-venv"
    echo "(다른 Python 버전을 쓰려면 PYTHON_BIN=python3.11 bash research/setup_ubuntu_env.sh 처럼 지정)"
    exit 1
fi

echo "=== 1. venv 생성 ==="
"$PYTHON_BIN" -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "=== 2. torch 설치 ==="
# CUDA 버전에 맞춘 정확한 명령은 https://pytorch.org/get-started/locally/ 에서 확인 가능.
# 렌더링 자체는 GPU 불필요 — CPU 전용이어도 아래 기본 설치로 충분.
pip install torch torchvision torchaudio

echo "=== 3. openvla-ivm 패키지 설치 ==="
pip install -e .

echo "=== 4. numpy<2 고정 (torch==2.2.0이 NumPy 2.0 이전 ABI) ==="
pip install "numpy<2"

echo "=== 5. LIBERO 설치 ==="
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

echo "=== 6. numpy<2 재고정 (LIBERO 관련 설치가 numpy를 다시 2.x로 올릴 수 있음) ==="
pip install "numpy<2"

echo ""
echo "=== 설치 완료 ==="
echo "확인 (venv 활성화된 상태에서):"
echo "  python research/data_generation/smoke_test.py --task_suite_name libero_spatial --task_id 0"
echo ""
echo "세그폴트/렌더링 에러가 나면 MUJOCO_GL=egl 또는 MUJOCO_GL=osmesa를 앞에 붙여서 재시도."
echo "(egl → osmesa → xvfb+glfw 순으로 시도, docs/MODEL.md 트러블슈팅 기록 참고)"
