#!/usr/bin/env python3
import dynamixel_sdk as dxl
import threading

# ── 설정 ──────────────────────────────────────────
BAUDRATE              = 1000000
PROTOCOL              = 2.0

ADDR_TORQUE_ENABLE    = 64
ADDR_PROFILE_VELOCITY = 112
LEN_PROFILE_VELOCITY  = 4

TORQUE_ON             = 1
TORQUE_OFF            = 0
PROFILE_VELOCITY      = 10

PORT_CONFIG = {
    '/dev/ttyUSB0': list(range(1, 10)),  # ID 1 ~ 9
    '/dev/ttyUSB1': list(range(1, 10)),  # ID 1 ~ 9
}
# ──────────────────────────────────────────────────

def set_profile_velocity(port, dxl_ids):
    portHandler   = dxl.PortHandler(port)
    packetHandler = dxl.PacketHandler(PROTOCOL)

    if not portHandler.openPort():
        print(f"[{port}] ❌ 포트 열기 실패")
        return
    if not portHandler.setBaudRate(BAUDRATE):
        print(f"[{port}] ❌ 보드레이트 설정 실패")
        return

    print(f"[{port}] ✅ 연결됨 | 담당 ID: {dxl_ids}")

    # ── Torque OFF ────────────────────────────────
    for dxl_id in dxl_ids:
        result, error = packetHandler.write1ByteTxRx(
            portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF
        )
        if result != dxl.COMM_SUCCESS:
            print(f"[{port}][ID {dxl_id}] Torque OFF 실패: {packetHandler.getTxRxResult(result)}")
        else:
            print(f"[{port}][ID {dxl_id}] Torque OFF ✅")

    # ── Sync Write ────────────────────────────────
    groupSyncWrite = dxl.GroupSyncWrite(
        portHandler, packetHandler,
        ADDR_PROFILE_VELOCITY, LEN_PROFILE_VELOCITY
    )

    added_ids = []
    for dxl_id in dxl_ids:
        param = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(PROFILE_VELOCITY)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(PROFILE_VELOCITY)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(PROFILE_VELOCITY)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(PROFILE_VELOCITY)),
        ]
        if groupSyncWrite.addParam(dxl_id, param):
            added_ids.append(dxl_id)
        else:
            print(f"[{port}][ID {dxl_id}] ❌ SyncWrite 파라미터 추가 실패 → 스킵")

    # addParam 성공한 ID가 있을 때만 전송
    if added_ids:
        result = groupSyncWrite.txPacket()
        if result != dxl.COMM_SUCCESS:
            print(f"[{port}] ❌ SyncWrite 전송 실패: {packetHandler.getTxRxResult(result)}")
        else:
            print(f"[{port}] ✅ Profile Velocity = {PROFILE_VELOCITY} 전송 완료 | ID: {added_ids}")
    else:
        print(f"[{port}] ❌ 전송할 ID 없음 (addParam 전부 실패)")

    groupSyncWrite.clearParam()

    # ── 확인 읽기 & Torque ON ─────────────────────
    for dxl_id in dxl_ids:
        val, result, error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PROFILE_VELOCITY
        )
        if result == dxl.COMM_SUCCESS:
            print(f"[{port}][ID {dxl_id}] Profile Velocity = {val} ✅")
        else:
            print(f"[{port}][ID {dxl_id}] ❌ 읽기 실패: {packetHandler.getTxRxResult(result)}")

        torque_result, error = packetHandler.write1ByteTxRx(
            portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON
        )
        if torque_result == dxl.COMM_SUCCESS:
            print(f"[{port}][ID {dxl_id}] Torque ON ✅")
        else:
            print(f"[{port}][ID {dxl_id}] ❌ Torque ON 실패: {packetHandler.getTxRxResult(torque_result)}")


    print(f"[{port}] ✅ Torque ON 완료")
    portHandler.closePort()

# ── 멀티스레드 실행 ───────────────────────────────
if __name__ == "__main__":
    threads = []
    for port, ids in PORT_CONFIG.items():
        t = threading.Thread(target=set_profile_velocity, args=(port, ids), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n[완료] 전체 포트 처리 종료")
