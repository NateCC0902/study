"""USV 航向保持模擬器（一階 Nomoto 模型）。

執行方式：  conda run -n study python usv_heading_sim_zh.py

情境：船一開始的航向（heading）為 0 度。在 t=0 時設定值（setpoint）為 90 度，
在 t=30s 時階躍到 180 度。一個常值擾動（disturbance；水流（current）／風（wind），作用起來像
卡住約 4 度的舵）整段時間都在推著船首（bow）——這正是 I 項必須配平
（trim）掉的東西。

在 PID.update() 尚未實作之前，本程式會以 NotImplementedError 結束。
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from pid_controller_zh import PID

# ---- 船體（Nomoto）參數 -------------------------------------------
K_NOMOTO = 0.6     # 轉向增益（turning gain）：每弧度舵角對應的穩態偏航角速度（rad/s）
T_NOMOTO = 2.5     # 時間常數（time constant，s）：偏航響應有多遲鈍
RUDDER_MAX = math.radians(35)   # 致動器飽和（saturation）
RUDDER_RATE = math.radians(25)  # 伺服機變化率限制（slew limit，rad/s）——見 notes 第 9 案
DISTURBANCE = math.radians(4)   # 水流／風，以等效卡住的舵角表示

# ---- 感測器 ---------------------------------------------------------------
NOISE_STD = 0.0    # 羅盤雜訊（compass noise），單位弧度。試試 math.radians(0.5) -> 觀察 D 項

# ---- 模擬 -----------------------------------------------------------------
DT = 0.05          # 20 Hz 控制器，業餘等級 USV 的典型值
T_END = 60.0


def wrap(angle):
    """將角度纏繞（wrap）到 (-pi, pi]。航向誤差一定要這樣做（notes 4.1）。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def setpoint_at(t):
    return math.radians(90) if t < 30.0 else math.radians(180)


def simulate(pid, seed=0):
    rng = np.random.default_rng(seed)
    n = int(T_END / DT)
    t = np.arange(n) * DT
    psi = np.zeros(n)      # 航向（rad）
    r = 0.0                # 偏航角速度（yaw rate，rad/s）
    rudder = np.zeros(n)
    sp = np.array([setpoint_at(ti) for ti in t])

    for i in range(1, n):
        measured = psi[i - 1] + rng.normal(0.0, NOISE_STD)
        error = wrap(sp[i] - measured)
        cmd = pid.update(error, DT)

        # 伺服機沒辦法瞬間移動：以 RUDDER_RATE 為上限朝命令值轉動
        step = RUDDER_RATE * DT
        rudder[i] = rudder[i - 1] + np.clip(cmd - rudder[i - 1], -step, step)

        # Nomoto：  T*r_dot + r = K*(rudder + disturbance)
        r_dot = (K_NOMOTO * (rudder[i] + DISTURBANCE) - r) / T_NOMOTO
        r += r_dot * DT
        psi[i] = psi[i - 1] + r * DT

    return t, np.degrees(psi), np.degrees(sp), np.degrees(rudder)


def main():
    # ---- 調這些增益 ------------------------------------------------------
    pid = PID(kp=2.0, ki=0.0, kd=0.0,
              output_limits=(-RUDDER_MAX, RUDDER_MAX))
    # ----------------------------------------------------------------------

    t, psi, sp, rudder = simulate(pid)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1.plot(t, sp, "k--", label="setpoint")
    ax1.plot(t, psi, label="heading")
    ax1.set_ylabel("heading (deg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, rudder, color="tab:orange")
    ax2.axhline(math.degrees(RUDDER_MAX), color="r", ls=":", lw=1)
    ax2.axhline(-math.degrees(RUDDER_MAX), color="r", ls=":", lw=1)
    ax2.set_ylabel("rudder (deg)")
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"kp={pid.kp}  ki={pid.ki}  kd={pid.kd}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# ---- 練習（在 update() 能運作之後做；案例編號對應 notes.zh-TW.md） -------
# 1. （第 1 案）只用 P，ki=kd=0：試試 kp = 0.5、2、5、8。你應該會在 kp=5 附近
#    發現持續振盪（sustained oscillation，那就是 Ku；週期 Tu 約 9 s）。把 kp
#    退回到約 2。
# 2. （第 2 案）在 kp=2 時，注意船會停在超過設定值約 2 度的位置——也就是
#    DISTURBANCE。試 kp=1 與 kp=4 來驗證偏差（offset）= 4/kp 度。
# 3. （第 3 案）加入 kd：先 1，再 3。找出剛好能消掉超調（overshoot）的 kd。
# 4. （第 4 案）加入 ki=0.2：偏差會在約 30 s 內消失。接著試 ki=2：緩慢的擺動。
# 5. （第 5 案）積分飽和（windup）：ki=1，把 setpoint_at() 改成永遠回傳 180 度。
#    先用你的抗積分飽和（anti-windup）跑一次，再暫時停掉它。會繞圈圈嗎？
# 6. （第 6 案）設定值在 t=30 s 階躍。如果你是對誤差（error）微分，舵命令會在
#    某一個取樣點猛然爆衝。改成對量測值微分，再比較看看。
# 7. （第 7 案）NOISE_STD = math.radians(0.5)：舵會不停地磨動。在 D 項上加一個
#    低通濾波器（low-pass filter，alpha 約 0.85），再比較。
