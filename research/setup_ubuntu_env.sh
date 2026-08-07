#!/bin/bash
# research/setup_ubuntu_env.sh
#
# Ubuntu(로컬/서버)에서 robosuite/LIBERO 데이터 생성 파이프라인을 돌리기 위한 conda 환경 설정.
#
# conda 사용 이유(2026-08-08 venv에서 전환): 원본 openvla, 그리고 우리가 fork한
# moojink/openvla-oft의 SETUP.md가 전부 conda 기반이라(`conda create -n ... python=3.10 -y` ->
# `conda activate` -> 이후 pip install) 선행 연구 계보와 통일하기 위함. 단, conda는 "격리된
# 파이썬 인터프리터 만들기" 용도로만 쓰고, 실제 패키지 설치는 원본과 동일하게 전부 pip로 한다
# (conda install/conda-forge 채널은 안 씀) — 그래서 아래 pip 레벨 버전 고정들은 venv 시절과
# 전부 동일하게 유지됨.
#
# 배경: Colab 컨테이너에 NVIDIA용 EGL 렌더링 드라이버(vendor ICD)가 없어서(Mesa만 있음),
# robosuite가 하드코딩해서 쓰는 EGL offscreen 렌더링이 계속 세그폴트로 죽었다. 렌더링
# 단계만 로컬 Ubuntu로 옮기기로 함 — VLA forward pass/LIVModule 학습은 계속 Colab에서.
# 실제 데스크톱 세션(진짜 디스플레이 있음)에서는 이 문제 자체가 없었음 — 확인 완료.
# 자세한 경위는 docs/MODEL.md 참고.
#
# 사용법: 이 저장소 루트에서 실행
#   bash research/setup_ubuntu_env.sh
#
# 완료 후:
#   conda activate openvla-ivm
#   echo "N" | python research/data_generation/smoke_test.py --task_suite_name libero_spatial --task_id 0
#   ("N"은 LIBERO가 최초 실행 시 물어보는 커스텀 데이터셋 경로 질문에 대한 자동 응답)

set -e

CONDA_ENV_NAME=${CONDA_ENV_NAME:-openvla-ivm}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

if ! command -v conda &> /dev/null; then
    echo "오류: conda를 찾을 수 없음."
    echo "설치: https://docs.conda.io/en/latest/miniconda.html (Miniconda로 충분)"
    exit 1
fi

echo "=== 1. conda 환경 생성 (${CONDA_ENV_NAME}, python ${PYTHON_VERSION}) ==="
# `conda activate`를 스크립트(비대화형 쉘) 안에서 쓰려면 conda의 쉘 훅을 먼저 로드해야 함.
eval "$(conda shell.bash hook)"
conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$CONDA_ENV_NAME"
pip install --upgrade pip

echo "=== 2. torch 설치 ==="
# CUDA 버전에 맞춘 정확한 명령은 https://pytorch.org/get-started/locally/ 에서 확인 가능.
# 렌더링 자체는 GPU 불필요 — CPU 전용이어도 아래 기본 설치로 충분.
pip install torch torchvision torchaudio

echo "=== 3. cmake 설치 (sentencepiece 등 일부 패키지가 소스 빌드 시 필요) ==="
# sudo 불필요 — PyPI의 cmake 바이너리 패키지를 conda 환경 안에 설치.
pip install cmake

echo "=== 4. openvla-ivm 패키지 설치 ==="
pip install -e .

echo "=== 5. numpy<2 고정 (torch==2.2.0이 NumPy 2.0 이전 ABI) ==="
pip install "numpy<2"

echo "=== 6. LIBERO 설치 ==="
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
# 주의: 기본(PEP 660) editable install은 LIBERO의 setup.py 구조를 못 읽어서
# MAPPING이 빈 채로 생성되고 `import libero`가 실패한다 (2026-07-30 확인).
# --config-settings editable_mode=compat (구 방식: easy-install.pth 기반)으로 우회.
pip install --config-settings editable_mode=compat -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

echo "=== 7. numpy<2 재고정 (LIBERO 관련 설치가 numpy를 다시 2.x로 올릴 수 있음) ==="
pip install "numpy<2"

echo "=== 8. scipy를 numpy<2 호환 버전으로 고정 ==="
# libero_requirements.txt가 최신 scipy(numpy>=2.0 요구)를 깔아서 numpy<2와 충돌 —
# libero_utils.py의 `import tensorflow` -> keras 레거시 경로가 scipy.sparse를 실제로 써서
# (Colab에서는 안 겪었던) AttributeError: module 'numpy' has no attribute 'long' 발생.
pip install "scipy<1.13"

echo "=== 9. mujoco를 robosuite==1.4.1과 호환되는 버전으로 고정 ==="
# libero_requirements.txt가 mujoco 버전을 안 박아서 최신(3.x 후반)이 깔리는데,
# robosuite==1.4.1의 컨트롤러 코드가 쓰는 MjData.qM이 최신 mujoco에서 제거됨
# (AttributeError: 'MjData' object has no attribute 'qM'). mujoco 3.0.0이 cp312 wheel이
# 있는 것 중 가장 오래된 버전이며 qM이 아직 남아있어 정상 동작 확인됨.
pip install "mujoco==3.0.0"

echo ""
echo "=== 설치 완료 ==="
echo "확인 (conda 환경 활성화된 상태에서):"
echo '  echo "N" | python research/data_generation/smoke_test.py --task_suite_name libero_spatial --task_id 0'
echo ""
echo "그래도 렌더링 관련 에러가 나면 (헤드리스/SSH 세션 등): MUJOCO_GL=egl 또는 osmesa 시도."
echo "(egl → osmesa → xvfb+glfw 순으로, docs/MODEL.md 트러블슈팅 기록 참고. 단, 실제 데스크톱"
echo " 세션에서는 이 문제 자체가 없었음 — 2026-07-30 로컬 Ubuntu에서 확인 완료)"
