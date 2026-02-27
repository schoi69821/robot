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
#VOLUME /root/cobot_magic_ws/magic-aloha-docker/cobot_magic_ws
 

#####3
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libgoogle-glog-dev \
    pkg-config \
    ros-noetic-image-geometry \
    ros-noetic-camera-info-manager \
    ros-noetic-image-transport \
    ros-noetic-image-publisher \
    libeigen3-dev \
    ros-noetic-backward-ros \
    libdw-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# libuvc 소스 빌드
RUN git clone https://github.com/libuvc/libuvc.git /tmp/libuvc && \
    cd /tmp/libuvc && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j4 && \
    make install && \
    ldconfig && \
    rm -rf /tmp/libuvc
    
# RealSense 
# librealsense 소스 코드 클론
RUN git clone https://github.com/IntelRealSense/librealsense.git /librealsense

# 소스 빌드
WORKDIR /librealsense
RUN mkdir build && cd build && \
    cmake ../ -DBUILD_EXAMPLES=true -DBUILD_WITH_CUDA=false && \
    make -j$(nproc) && \
    make install
    
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y ros-noetic-ros-base ros-noetic-realsense2-camera && \
    rm -rf /var/lib/apt/lists/*
#####
WORKDIR /root
# 1. 디렉토리 생성 및 cudnn 설치
RUN mkdir -p /root/miniconda3 && \
    mkdir -p /root/cudnn && \
    test -f /root/cobot_magic_ws/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz && \
    cp /root/cobot_magic_ws/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz /root/cudnn/ && \
    tar xvf /root/cudnn/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz -C /root/cudnn/ && \
    cp -P /root/cudnn/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/* /usr/local/cuda/include/ && \
    cp -P /root/cudnn/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/* /usr/local/cuda/lib64/

# 2. Miniconda 설치 (수정된 버전)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh

# conda 환경 설정
#ENV PATH="/root/miniconda3/bin:${PATH}"
#RUN conda init bash && \
#    conda create -n aloha python=3.8 -y && \
#    conda install -n aloha numpy -y

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash" # && catkin_init_workspace"
# Install package dependencies
#RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y
# Build the workspace

WORKDIR /root/cobot_magic_ws/cobot_magic_two/camera_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
    
WORKDIR /root/cobot_magic_ws/cobot_magic_two/piper_ros_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

WORKDIR /root
RUN /bin/bash -c "source /root/cobot_magic_ws/cobot_magic_two/camera_ws/devel/setup.bash"
RUN /bin/bash -c "source /root/cobot_magic_ws/cobot_magic_two/piper_ros_ws/devel/setup.bash"

RUN apt-get update && apt-get install -y tmux

# 7) 프로젝트별 의존성 설치
WORKDIR /root/cobot_magic_ws/cobot_magic_two/collect_data
RUN pip3 install --ignore-installed --no-cache-dir -r requirements.txt


WORKDIR /root/cobot_magic_ws/cobot_magic_two/aloha-devel
RUN pip3 install --ignore-installed --no-cache-dir -r requirements.txt

WORKDIR /root/cobot_magic_ws/cobot_magic_two/aloha-devel/act/detr
RUN pip3 install -v -e .
RUN chmod a+r /usr/local/cuda/lib64/* && \  
    rm -rf /root/cudnn #&& \


# 8) 최종 작업 디렉토리 & CMD
WORKDIR /root/cobot_magic_ws
CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && bash"]

