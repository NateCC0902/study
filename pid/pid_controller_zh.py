"""一個 PID 控制器，為了理解而打造——之後也將用於 USV。

實作前請先閱讀 notes.zh-TW.md 的第 2 節與第 4 節。
"""


class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min, self.out_max = output_limits

        # 控制器（controller）狀態（會在多次呼叫之間改變）
        self.integral = 0.0
        self.prev_error = None  # None 代表「尚未有前一筆取樣點（sample）」

    def reset(self):
        """當控制器在關閉／被人為接管之後重新接上時呼叫。"""
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        """一個控制步。回傳致動器（actuator）命令 u。

        error : setpoint - measurement，亦即設定值（setpoint）減去量測值（measurement），
                且已完成角度纏繞（angle wrapping，單位為弧度 radians）
        dt    : 距上一次呼叫所經過的秒數（> 0）

        TODO(你)——實作離散形式的 PID 控制律（control law）:
          1. P 項: kp * error
          2. I 項: 累加 `self.integral`，再乘上 ki
          3. D 項: 誤差（error）的斜率，(error - prev_error) / dt。
             在最一開始的那一次呼叫時 prev_error 為 None——此時微分（derivative）
             無從定義，因此這一筆取樣點就用 0 代替。
          4. 把各項相加，箝位（clamp）到 [self.out_min, self.out_max]，再回傳。

        設計抉擇（由你決定）: 抗積分飽和（anti-windup）。如果你只做步驟 1-4，
        當輸出被釘在上下限時，積分（integral）量仍會持續增長
        （見 notes.zh-TW.md 4.2）。請挑一種策略:
          a) 箝位 self.integral，使得 ki*integral 不會超出輸出範圍
          b) 條件積分（conditional integration）: 在輸出已飽和（saturation）的取樣點上，跳過該次積分
          c) 反算法（back-calculation）: integral -= (u_unclamped - u_clamped) * gain
        (a) 通常是第一版的實作做法。之後你可以在模擬器（simulator）裡
        比較各種策略——練習 4。
        """
        raise NotImplementedError("把我實作出來，然後執行 usv_heading_sim_zh.py")
