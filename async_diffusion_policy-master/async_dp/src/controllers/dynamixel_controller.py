"""
Dynamixel Robot Controller
Ready-to-use controller for Dynamixel-based robots (VX300s, etc.)
"""
import math
import numpy as np
import time
import logging
from typing import Optional, List
from dataclasses import dataclass, field

from src.controllers.base_controller import BaseRobotController, ControllerConfig
from src.interfaces import InterfaceConfig

logger = logging.getLogger(__name__)

# Try to import Dynamixel SDK
try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler,
        GroupSyncRead, GroupSyncWrite,
        COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
        DXL_LOWORD, DXL_HIWORD
    )
    DYNAMIXEL_AVAILABLE = True
except ImportError:
    DYNAMIXEL_AVAILABLE = False
    logger.warning("[Dynamixel] SDK not installed. Install with: pip install dynamixel-sdk")


# Dynamixel Protocol 2.0 Control Table Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108

# Data lengths
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4

# Protocol version
PROTOCOL_VERSION = 2.0


@dataclass
class DynamixelConfig(ControllerConfig):
    """Dynamixel-specific configuration"""
    # Port settings
    port: str = "/dev/ttyUSB0"  # Linux: /dev/ttyUSB0, Windows: COM3
    baudrate: int = 1000000     # 1Mbps

    # Motor IDs
    motor_ids: List[int] = field(default_factory=lambda: list(range(1, 8)))  # [1,2,3,4,5,6,7]

    # Position limits (in Dynamixel units, 0-4095)
    position_min: int = 0
    position_max: int = 4095

    # Conversion factors
    position_to_rad: float = 2 * math.pi / 4096  # 4096 units = 2*pi rad
    velocity_to_rad_s: float = 0.229 * (2 * math.pi / 60)  # rpm to rad/s

    # Motion profile
    profile_velocity: int = 100      # 0-32767
    profile_acceleration: int = 50   # 0-32767


class DynamixelController(BaseRobotController):
    """
    Controller for Dynamixel servo motors.

    Supports:
        - Dynamixel X-series (XM430, XM540, etc.)
        - Dynamixel Pro series
        - VX300s robot arm (Interbotix)

    Usage:
        config = DynamixelConfig(
            port="/dev/ttyUSB0",
            motor_ids=[1, 2, 3, 4, 5, 6],
            interface_config=InterfaceConfig(
                action_dim=6,
                obs_dim=6,
                shm_name="vx300s_robot"
            )
        )

        controller = DynamixelController(config)
        controller.start()

        # Controller runs in background thread
        # Async DP connects to SharedMemory interface

        input("Press Enter to stop...")
        controller.stop()
    """

    def __init__(self, config: DynamixelConfig):
        # Update interface config dimensions based on motor count
        config.interface_config.action_dim = len(config.motor_ids)
        config.interface_config.obs_dim = len(config.motor_ids)

        super().__init__(config)
        self.dxl_config = config

        # Dynamixel SDK objects
        self._port_handler = None
        self._packet_handler = None
        self._group_sync_read_pos = None
        self._group_sync_read_vel = None
        self._group_sync_write = None

        # Simulation mode flag
        self._simulation_mode = not DYNAMIXEL_AVAILABLE

    def _connect_hardware(self) -> bool:
        """Connect to Dynamixel motors."""
        if self._simulation_mode:
            logger.info("[Dynamixel] Running in SIMULATION mode (SDK not available)")
            return True

        try:
            # Initialize port handler
            self._port_handler = PortHandler(self.dxl_config.port)

            if not self._port_handler.openPort():
                logger.error(f"[Dynamixel] Failed to open port: {self.dxl_config.port}")
                return False

            if not self._port_handler.setBaudRate(self.dxl_config.baudrate):
                logger.error(f"[Dynamixel] Failed to set baudrate: {self.dxl_config.baudrate}")
                return False

            logger.info(f"[Dynamixel] Port opened: {self.dxl_config.port} @ {self.dxl_config.baudrate}")

            # Initialize packet handler
            self._packet_handler = PacketHandler(PROTOCOL_VERSION)

            # Initialize sync read/write
            self._init_sync_handlers()

            # Enable torque and set motion profile
            self._setup_motors()

            logger.info(f"[Dynamixel] Connected to {len(self.dxl_config.motor_ids)} motors")
            return True

        except Exception as e:
            logger.error(f"[Dynamixel] Connection error: {e}")
            return False

    def _init_sync_handlers(self):
        """Initialize sync read/write handlers."""
        # Sync read for position
        self._group_sync_read_pos = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION
        )

        # Sync read for velocity
        self._group_sync_read_vel = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY
        )

        # Sync write for goal position
        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )

        # Add motors to sync read
        for motor_id in self.dxl_config.motor_ids:
            self._group_sync_read_pos.addParam(motor_id)
            self._group_sync_read_vel.addParam(motor_id)

    def _setup_motors(self):
        """Setup motor parameters.

        Dynamixel requires torque OFF before changing operating mode.
        Sequence: Torque OFF -> Set mode -> Set profile -> Torque ON
        """
        for motor_id in self.dxl_config.motor_ids:
            # Torque OFF first (required before changing operating mode)
            self._write_register(motor_id, ADDR_TORQUE_ENABLE, 0, 1)

            # Set operating mode to position control (3)
            self._write_register(motor_id, ADDR_OPERATING_MODE, 3, 1)

            # Set motion profile
            self._write_register(motor_id, ADDR_PROFILE_VELOCITY,
                               self.dxl_config.profile_velocity, 4)
            self._write_register(motor_id, ADDR_PROFILE_ACCELERATION,
                               self.dxl_config.profile_acceleration, 4)

            # Enable torque
            self._write_register(motor_id, ADDR_TORQUE_ENABLE, 1, 1)

    def _write_register(self, motor_id: int, address: int, value: int, length: int):
        """Write to motor register with result checking."""
        if length == 1:
            result, error = self._packet_handler.write1ByteTxRx(
                self._port_handler, motor_id, address, value
            )
        elif length == 2:
            result, error = self._packet_handler.write2ByteTxRx(
                self._port_handler, motor_id, address, value
            )
        elif length == 4:
            result, error = self._packet_handler.write4ByteTxRx(
                self._port_handler, motor_id, address, value
            )
        else:
            logger.error(f"[Dynamixel] Invalid write length: {length}")
            return

        if result != COMM_SUCCESS:
            logger.warning(f"[Dynamixel] Write failed ID={motor_id} addr={address}: "
                          f"{self._packet_handler.getTxRxResult(result)}")
        elif error != 0:
            logger.warning(f"[Dynamixel] Hardware error ID={motor_id} addr={address}: "
                          f"{self._packet_handler.getRxPacketError(error)}")

    def _disconnect_hardware(self) -> None:
        """Disconnect from Dynamixel motors."""
        if self._simulation_mode:
            return

        try:
            # Disable torque
            if self._packet_handler and self._port_handler:
                for motor_id in self.dxl_config.motor_ids:
                    self._write_register(motor_id, ADDR_TORQUE_ENABLE, 0, 1)

            # Close port
            if self._port_handler:
                self._port_handler.closePort()

            logger.info("[Dynamixel] Disconnected")

        except Exception as e:
            logger.error(f"[Dynamixel] Disconnect error: {e}")

    def _read_state(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Read current motor positions and velocities."""
        num_motors = len(self.dxl_config.motor_ids)

        if self._simulation_mode:
            # Return simulated state
            return (
                self._current_qpos.copy(),
                np.zeros(num_motors, dtype=np.float32)
            )

        positions = np.zeros(num_motors, dtype=np.float32)
        velocities = np.zeros(num_motors, dtype=np.float32)

        try:
            # Read positions
            result = self._group_sync_read_pos.txRxPacket()
            if result == COMM_SUCCESS:
                for i, motor_id in enumerate(self.dxl_config.motor_ids):
                    if self._group_sync_read_pos.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                        raw_pos = self._group_sync_read_pos.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                        # Convert to radians (center at 2048)
                        positions[i] = (raw_pos - 2048) * self.dxl_config.position_to_rad

            # Read velocities
            result = self._group_sync_read_vel.txRxPacket()
            if result == COMM_SUCCESS:
                for i, motor_id in enumerate(self.dxl_config.motor_ids):
                    if self._group_sync_read_vel.isAvailable(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                        raw_vel = self._group_sync_read_vel.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                        # Handle signed velocity
                        if raw_vel > 0x7FFFFFFF:
                            raw_vel -= 0x100000000
                        velocities[i] = raw_vel * self.dxl_config.velocity_to_rad_s

        except Exception as e:
            logger.error(f"[Dynamixel] Read error: {e}")

        return positions, velocities

    def _write_command(self, action: np.ndarray) -> bool:
        """Write position command to motors."""
        if self._simulation_mode:
            # Update simulated state
            self._current_qpos = action.copy()
            return True

        try:
            # Clear previous parameters
            self._group_sync_write.clearParam()

            for i, motor_id in enumerate(self.dxl_config.motor_ids):
                # Convert radians to Dynamixel units
                raw_pos = int(action[i] / self.dxl_config.position_to_rad + 2048)

                # Clamp to valid range
                raw_pos = int(np.clip(
                    raw_pos,
                    self.dxl_config.position_min,
                    self.dxl_config.position_max
                ))

                # Pack position data
                param = [
                    DXL_LOBYTE(DXL_LOWORD(raw_pos)),
                    DXL_HIBYTE(DXL_LOWORD(raw_pos)),
                    DXL_LOBYTE(DXL_HIWORD(raw_pos)),
                    DXL_HIBYTE(DXL_HIWORD(raw_pos))
                ]

                self._group_sync_write.addParam(motor_id, param)

            # Send command
            result = self._group_sync_write.txPacket()
            return result == COMM_SUCCESS

        except Exception as e:
            logger.error(f"[Dynamixel] Write error: {e}")
            return False

    def enable_torque(self, enable: bool = True) -> None:
        """Enable or disable motor torque."""
        if self._simulation_mode:
            return

        value = 1 if enable else 0
        for motor_id in self.dxl_config.motor_ids:
            self._write_register(motor_id, ADDR_TORQUE_ENABLE, value, 1)

        logger.info(f"[Dynamixel] Torque {'enabled' if enable else 'disabled'}")

    def set_profile(self, velocity: int, acceleration: int) -> None:
        """Set motion profile for all motors."""
        if self._simulation_mode:
            return

        for motor_id in self.dxl_config.motor_ids:
            self._write_register(motor_id, ADDR_PROFILE_VELOCITY, velocity, 4)
            self._write_register(motor_id, ADDR_PROFILE_ACCELERATION, acceleration, 4)

        logger.info(f"[Dynamixel] Profile set: vel={velocity}, acc={acceleration}")

    def go_home(self, home_position: Optional[np.ndarray] = None) -> None:
        """Move robot to home position."""
        if home_position is None:
            home_position = np.zeros(len(self.dxl_config.motor_ids), dtype=np.float32)

        logger.info("[Dynamixel] Moving to home position...")
        self._write_command(home_position)
        time.sleep(2.0)  # Wait for motion to complete


# =============================================================================
# Convenience functions
# =============================================================================

def create_vx300s_controller(
    port: str = "/dev/ttyUSB0",
    shm_name: str = "vx300s_robot"
) -> DynamixelController:
    """
    Create controller for Interbotix VX300s robot arm.

    Args:
        port: Serial port (Linux: /dev/ttyUSB0, Windows: COM3)
        shm_name: Shared memory name for Async DP interface

    Returns:
        Configured DynamixelController
    """
    config = DynamixelConfig(
        port=port,
        motor_ids=[1, 2, 3, 4, 5, 6],  # 6 DOF arm
        baudrate=1000000,
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=6,
            obs_dim=6,
            shm_name=shm_name
        )
    )
    return DynamixelController(config)


def create_dual_arm_controller(
    port: str = "/dev/ttyUSB0",
    shm_name: str = "dual_arm_robot"
) -> DynamixelController:
    """
    Create controller for dual-arm robot (14 DOF).

    Args:
        port: Serial port
        shm_name: Shared memory name

    Returns:
        Configured DynamixelController
    """
    config = DynamixelConfig(
        port=port,
        motor_ids=list(range(1, 15)),  # 14 motors
        baudrate=1000000,
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=14,
            obs_dim=14,
            shm_name=shm_name
        )
    )
    return DynamixelController(config)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 60)
    print("  Dynamixel Controller Example")
    print("=" * 60)

    # Check if running on Windows
    if sys.platform == 'win32':
        port = "COM3"
    else:
        port = "/dev/ttyUSB0"

    # Create controller
    controller = create_vx300s_controller(port=port, shm_name="example_robot")

    # Set callbacks
    def on_state_change(old, new):
        print(f"State: {old.value} -> {new.value}")

    def on_error(e):
        print(f"Error: {e}")

    controller.set_callbacks(on_state_change=on_state_change, on_error=on_error)

    # Start controller
    if not controller.start():
        print("Failed to start controller!")
        sys.exit(1)

    print("\nController running. Press Ctrl+C to stop...")
    print("Connect Async DP to shared memory 'example_robot'\n")

    try:
        while True:
            # Print statistics every 5 seconds
            stats = controller.get_statistics()
            print(f"Loops: {stats['loop_count']}, "
                  f"Avg: {stats['avg_loop_time_ms']:.2f}ms, "
                  f"qpos[0]: {stats['current_qpos'][0]:.3f}")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\nStopping...")

    controller.stop()
    print("Done.")
