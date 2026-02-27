# 1) CUDA 베이스
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2) 필수 툴
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# 3) ROS 1 Noetic 저장소 등록 및 설치
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
 && apt-get update \
 && apt-get install -y ros-noetic-desktop-full python3-rosdep libx11-6 libxrender1 libxcb1 libxkbcommon0 usbutils can-utils ethtool iproute2 \
 && rm -rf /var/lib/apt/lists/*

# rosdep 초기화 & 환경 자동 로드
RUN rosdep init && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# 4) pip 업그레이드 
RUN pip3 install --upgrade pip
RUN pip3 install huggingface_hub==0.25 

RUN pip3 install python-can piper_sdk scipy

# 5) PyTorch GPU 버전
RUN pip3 install \
      torch==2.1.1 \
      torchvision==0.16.1 \
      torchaudio==2.1.1 \
      --index-url https://download.pytorch.org/whl/cu118

# 6) 워크스페이스 복사
WORKDIR /root
COPY cobot_magic_ws /root/cobot_magic_ws

# 7) 프로젝트별 의존성 설치
WORKDIR /root/cobot_magic_ws/cobot_magic_four/collect_data
RUN pip3 install --ignore-installed --no-cache-dir -r requirements.txt

WORKDIR /root/cobot_magic_ws/cobot_magic_four/aloha-devel
RUN pip3 install --ignore-installed --no-cache-dir -r requirements.txt

WORKDIR /root/cobot_magic_ws/cobot_magic_four/aloha-devel/act/detr
RUN pip3 install -v -e .

# 8) 최종 작업 디렉토리 & CMD
WORKDIR /root/cobot_magic_ws
CMD ["bash"]

