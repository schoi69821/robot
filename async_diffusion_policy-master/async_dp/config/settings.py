import numpy as np
import os

class Config:
    PROJECT_NAME = "Async DP Wafer Inspector"
    SHM_NAME = "asyncdp_shm_v1"
    
    # Hardware Config
    ROBOT_MODEL = 'vx300s'
    CONTROL_FREQ = 500  # Hz (Muscle)
    GRIPPER_FORCE = 150 # mA
    
    # AI Model Config
    INFERENCE_FREQ = 15 # Hz (Brain)
    PRED_HORIZON = 16   # Steps
    OBS_HORIZON = 2     # Steps
    ACTION_DIM = 14     # 7 DOF x 2 Arms (or 6+1 gripper)
    OBS_DIM = 14
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "assets/data")
    CKPT_PATH = os.path.join(BASE_DIR, "assets/checkpoints/best_model.pth")
    
    # Safety Limits
    MAX_JOINT_VEL = 2.0
    EMERGENCY_STOP_LIMIT = 3.1