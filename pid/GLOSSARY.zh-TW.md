# PID 控制教學模組 — 中英術語對照表（臺灣正體 zh-TW）

本表為本模組所有譯者的權威詞彙基準。請對每個英文術語使用「繁體中文」欄位指定的「唯一」中文譯名，務必前後一致。
「首次出現寫法」欄給出該術語在文中第一次出現時應採用的「中文（English）」格式；之後再次出現時，單用中文即可。
凡屬數學符號、程式識別字、單位、專有名詞者，一律保留原文不譯（見備註）。

| English | 繁體中文 | 首次出現寫法 (中文（English）) | 備註 |
|---|---|---|---|
| PID | PID | PID | 專有縮寫，保留原文；可在首段補述為「比例—積分—微分（PID）控制器」。 |
| controller | 控制器 | 控制器（controller） | — |
| proportional | 比例 | 比例（proportional） | P 項。 |
| integral | 積分 | 積分（integral） | I 項；亦指「積分量」狀態變數。 |
| derivative | 微分 | 微分（derivative） | D 項。 |
| P term | P 項 | P 項 | 符號 P 保留原文。 |
| I term | I 項 | I 項 | 符號 I 保留原文。 |
| D term | D 項 | D 項 | 符號 D 保留原文。 |
| gain | 增益 | 增益（gain） | — |
| kp | kp | kp | 程式識別字／符號，保留原文。比例增益。 |
| ki | ki | ki | 程式識別字／符號，保留原文。積分增益。 |
| kd | kd | kd | 程式識別字／符號，保留原文。微分增益。 |
| proportional gain | 比例增益 | 比例增益（proportional gain） | 即 kp。 |
| integral gain | 積分增益 | 積分增益（integral gain） | 即 ki。 |
| derivative gain | 微分增益 | 微分增益（derivative gain） | 即 kd。 |
| setpoint | 設定值 | 設定值（setpoint） | 程式中變數 setpoint／sp 保留原文。 |
| error | 誤差 | 誤差（error） | 符號 e、變數 error 保留原文。 |
| measurement | 量測值 | 量測值（measurement） | — |
| heading | 航向 | 航向（heading） | 符號 ψ／變數 psi 保留原文。 |
| yaw | 偏航 | 偏航（yaw） | — |
| yaw rate | 偏航角速度 | 偏航角速度（yaw rate） | 符號 r 保留原文。 |
| rudder | 舵 | 舵（rudder） | 符號 δ、變數 rudder 保留原文。 |
| rudder angle | 舵角 | 舵角（rudder angle） | — |
| rudder stop | 舵止／舵極限 | 舵止（rudder stop） | 指機械硬限位（±35°）。 |
| servo | 伺服機 | 伺服機（servo） | 亦可單稱「伺服」。 |
| actuator | 致動器 | 致動器（actuator） | — |
| slew rate | 變化率限制 | 變化率限制（slew rate） | 指伺服機的最大轉動速率。 |
| rate limit | 速率限制 | 速率限制（rate limit） | 與 slew rate 同義，統一以「速率限制」對應 rate limit。 |
| saturation | 飽和 | 飽和（saturation） | 致動器輸出達上下限。 |
| clamp | 箝位 | 箝位（clamp） | 動詞「將……箝制在上下限內」；程式函式 clamp 保留原文。 |
| output limits | 輸出限制 | 輸出限制（output limits） | 程式參數 output_limits 保留原文。 |
| disturbance | 擾動 | 擾動（disturbance） | 變數 DISTURBANCE／DIST 保留原文。 |
| current | 水流 | 水流（current） | 海流／水流，推動船首的擾動之一。 |
| wind | 風 | 風（wind） | 擾動之一。 |
| Nomoto model | Nomoto 模型 | Nomoto 模型（Nomoto model） | 專有名詞 Nomoto 保留原文。 |
| plant | 受控體 | 受控體（plant） | 控制理論中被控制的對象（此處即船體模型）。 |
| plant model | 受控體模型 | 受控體模型（plant model） | — |
| hull | 船體 | 船體（hull） | — |
| bow | 船首 | 船首（bow） | — |
| time constant | 時間常數 | 時間常數（time constant） | 符號 T 保留原文。 |
| turning gain | 轉向增益 | 轉向增益（turning gain） | Nomoto 的 K，符號 K 保留原文。 |
| control loop | 控制迴圈 | 控制迴圈（control loop） | — |
| closed loop | 閉迴路 | 閉迴路（closed loop） | — |
| open loop | 開迴路 | 開迴路（open loop） | — |
| control law | 控制律 | 控制律（control law） | — |
| feedback | 回授 | 回授（feedback） | 切勿用「回饋」。 |
| timestep | 時間步 | 時間步（timestep） | — |
| dt | dt | dt | 程式識別字／符號，保留原文。每步時間間隔。 |
| sampling | 取樣 | 取樣（sampling） | — |
| sample | 取樣點 | 取樣點（sample） | 視語境亦可譯「一次取樣」。 |
| sampling rate | 取樣率 | 取樣率（sampling rate） | — |
| update rate | 更新率 | 更新率（update rate） | 控制迴圈頻率（如 20 Hz）。 |
| overshoot | 超調 | 超調（overshoot） | 亦可寫「過衝」，本模組統一用「超調」。 |
| oscillation | 振盪 | 振盪（oscillation） | — |
| sustained oscillation | 持續振盪 | 持續振盪（sustained oscillation） | — |
| ringing | 振鈴 | 振鈴（ringing） | 小幅衰減振盪；與 oscillation 區分。 |
| ultimate gain | 臨界增益 | 臨界增益（ultimate gain, Ku） | 符號 Ku 保留原文。 |
| Ku | Ku | Ku | 符號保留原文。臨界增益。 |
| period | 週期 | 週期（period） | — |
| ultimate period | 臨界週期 | 臨界週期（ultimate period, Tu） | 符號 Tu 保留原文。 |
| Tu | Tu | Tu | 符號保留原文。臨界週期。 |
| Ziegler–Nichols | Ziegler–Nichols | Ziegler–Nichols | 專有名詞，保留原文；可註明「齊格勒—尼可斯整定法」。 |
| tuning | 整定 | 整定（tuning） | 調整增益的程序；亦可用「調校」。 |
| de-tune | 退調 | 退調（de-tune） | 將過熱的增益往保守方向回調。 |
| steady-state offset | 穩態偏差 | 穩態偏差（steady-state offset） | 符號 e_ss 保留原文。 |
| steady-state error | 穩態誤差 | 穩態誤差（steady-state error） | — |
| offset | 偏差 | 偏差（offset） | 與 steady-state offset 一致。 |
| disturbance rejection | 擾動抑制 | 擾動抑制（disturbance rejection） | — |
| damping | 阻尼 | 阻尼（damping） | — |
| spring | 彈簧 | 彈簧（spring） | P 項的類比。 |
| damper | 阻尼器 | 阻尼器（damper） | D 項的類比。 |
| mass | 質量 | 質量（mass） | 彈簧—質量類比。 |
| inertia | 慣性 | 慣性（inertia） | 此處指轉動慣性。 |
| trim tab | 配平片 | 配平片（trim tab） | I 項的類比；亦稱「微調片」。 |
| trim | 配平 | 配平（trim） | 動詞「修整／平衡掉持續擾動」。 |
| windup | 積分飽和 | 積分飽和（積分捲繞，windup） | 統一以「積分飽和」為主譯名，可附「積分捲繞」。 |
| anti-windup | 抗積分飽和 | 抗積分飽和（anti-windup） | — |
| conditional integration | 條件積分 | 條件積分（conditional integration） | 抗積分飽和策略之一。 |
| back-calculation | 反算法 | 反算法（back-calculation） | 抗積分飽和策略之一。 |
| derivative kick | 微分衝擊 | 微分衝擊（derivative kick） | — |
| derivative-on-measurement | 對量測值微分 | 對量測值微分（derivative-on-measurement） | 避免微分衝擊的做法。 |
| derivative-on-error | 對誤差微分 | 對誤差微分（derivative-on-error） | 會引發微分衝擊的做法。 |
| noise | 雜訊 | 雜訊（noise） | 切勿用「噪聲」。 |
| compass noise | 羅盤雜訊 | 羅盤雜訊（compass noise） | — |
| low-pass filter | 低通濾波器 | 低通濾波器（low-pass filter） | — |
| filter | 濾波器 | 濾波器（filter） | — |
| alpha | alpha | alpha（α） | 濾波係數；符號 α 與變數 alpha 保留原文。 |
| angle wrapping | 角度纏繞 | 角度纏繞（angle wrapping） | 程式函式 wrap 保留原文。 |
| wrap | 纏繞 | 纏繞（wrap） | 動詞，將角度折回 (-π, π]；函式名保留原文。 |
| waypoint | 航點 | 航點（waypoint） | — |
| differential thrust | 差動推力 | 差動推力（differential thrust） | 以左右推進器差速轉向。 |
| thruster | 推進器 | 推進器（thruster） | — |
| kill switch | 緊急斷電開關 | 緊急斷電開關（kill switch） | 硬體安全開關。 |
| IMU | IMU | IMU | 慣性量測單元，專有縮寫保留原文。 |
| EKF | EKF | EKF | 擴展卡爾曼濾波器，專有縮寫保留原文。 |
| complementary filter | 互補濾波器 | 互補濾波器（complementary filter） | — |
| sensor fusion | 感測器融合 | 感測器融合（sensor fusion） | 融合陀螺儀與羅盤。 |
| gyro | 陀螺儀 | 陀螺儀（gyro） | — |
| phase lag | 相位延遲 | 相位延遲（phase lag） | — |
| lag | 延遲 | 延遲（lag） | 一般性「落後／遲滯」；視語境亦可「遲滯」。 |
| latency | 延遲時間 | 延遲時間（latency） | 通訊延遲。 |
| limit cycle | 極限環 | 極限環（limit cycle） | 持續的等幅振盪。 |
| response | 響應 | 響應（response） | — |
| first-order | 一階 | 一階（first-order） | 一階系統（Nomoto）。 |
| USV | USV | USV | 無人水面載具，專有縮寫保留原文。 |
| UAV | UAV | UAV | 無人飛行載具，專有縮寫保留原文。 |
| ROV | ROV | ROV | 遙控潛水載具，專有縮寫保留原文。 |
| autopilot | 自動駕駛儀 | 自動駕駛儀（autopilot） | — |
| autonomy | 自主航行 | 自主航行（autonomy） | — |
| ROS2 | ROS2 | ROS2 | 專有名詞，保留原文。 |
| node | 節點 | 節點（node） | ROS2 節點。 |
| timer callback | 計時器回呼 | 計時器回呼（timer callback） | — |
| odometry | 里程計 | 里程計（odometry） | 對應 /odometry/filtered，路徑保留原文。 |
| flight controller | 飛控板 | 飛控板（flight controller） | — |
| reset | 重置 | 重置（reset） | 程式方法 reset() 保留原文。 |
| update | 更新 | 更新（update） | 程式方法 update() 保留原文。 |
| simulator | 模擬器 | 模擬器（simulator） | — |
| simulation | 模擬 | 模擬（simulation） | — |
| figure | 圖 | 圖（figure） | 圖表。 |
| panel | 子圖 | 子圖（panel） | 圖中的上／下分格。 |
| sweep | 掃描 | 掃描（sweep） | 增益掃描（逐一試不同 kp 等）。 |
