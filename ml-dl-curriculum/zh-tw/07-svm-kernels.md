# 07 — 支持向量機與核

> 第 2 部分 · 第 07 課 · 程式技術棧：scikit-learn

**先備知識：** [06 — k-NN、決策樹與集成](06-knn-trees-ensembles.md)

**學完本課你能：**
- 解釋*為什麼***最大間隔 (maximum-margin)** 超平面是「最安全」的線性分隔器，並指認出定義它的**支持向量 (support vector)**。
- 把支持向量機的目標函數連結到**鉸鏈損失 (hinge loss)** + L2 正則化，並調校**軟間隔 C** 參數。
- 運用**核技巧 (kernel trick)** 在不真正計算高維特徵的情況下分隔非線性資料，在**線性 / 多項式 / RBF** 核之間做明智選擇，並調校 **gamma**。
- 在 scikit-learn 中訓練並視覺化 `SVC` 的決策邊界，並判斷何時支持向量機勝過樹集成。

---

## 1. 直覺理解

邏輯迴歸的邊界（第 04 課）只要點落在正確的一側就心滿意足了。兩條都能完美分隔資料的不同邊界，在它眼中看起來一樣好。但它們*並不*一樣好——其中一條可能擦著你最近的資料點只差一根頭髮的距離，另一條則可能正好把間隙從正中央劈開。

**支持向量機 (Support Vector Machine, SVM)** 挑的是那條坐落在兩類別之間*最寬的空街正中央*的線。想像在兩個群集之間的縫隙裡推出一條盡可能最胖的道路：**間隔 (margin)** 就是那條路的寬度，而邊界就是它的中線。觸碰到路緣的那少數幾個點就是**支持向量**——只有它們決定了這條路。其他任何點都可以刪掉，邊界也不會移動。

```mermaid
flowchart LR
    A["許多條分隔線<br/>都能擬合資料"] --> B["SVM 規則：<br/>最大化間隔<br/>（最寬的街）"]
    B --> C["邊界只由<br/>路緣上的支持向量決定"]
    C --> D["資料無法線性分隔？"]
    D -->|"透過核<br/>提升到更高維"| E["那裡是一條直街<br/>= 這裡是一條彎曲邊界"]
```

為什麼最寬的街是聰明的選擇？因為它對雜訊最*穩健*。一個落點稍微偏離訓練點的新測量值仍會落在正確的一側，因為你留下了最大的緩衝。這是一種內建的正則化形式——大間隔正是第 05 課「保持模型簡單」在幾何上的表親。

第二個重要概念：真實感測器資料很少能被一條直線分隔。**核技巧**讓支持向量機表現得彷彿把你的點提升到一個維度高得多的空間，在那裡一個平面*確實*能分隔它們——而且完全不必付出計算那些座標的代價。你 2D 世界裡的一條彎曲邊界，不過是那個隱藏高維世界裡的一刀直切。

---

## 2. 數學原理

### 分隔超平面

線性分類器是一個超平面

$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$$

- $\mathbf{x} \in \mathbb{R}^d$ —— 一個特徵向量（例如 $d$ 個感測器讀值）。
- $\mathbf{w} \in \mathbb{R}^d$ —— 權重向量，**法線（垂直於邊界）**。
- $b \in \mathbb{R}$ —— 偏值 / 位移，把平面從原點推開。

我們以 $\hat{y} = \operatorname{sign}(f(\mathbf{x}))$ 預測類別，標籤編碼為 $y \in \{-1, +1\}$（注意：支持向量機用 $\pm 1$，而非 $0/1$）。

### 間隔 = 到最近點的距離

點 $\mathbf{x}$ 到超平面的有號距離是 $f(\mathbf{x}) / \lVert \mathbf{w} \rVert$（第 01 課的標準點到平面公式）。我們可以隨意重新縮放 $\mathbf{w}, b$，所以我們透過要求最近的點滿足 $|f(\mathbf{x})| = 1$ 來固定尺度。那些點坐落在**間隔邊界** $f(\mathbf{x}) = \pm 1$ 上，而這條街的半寬是

$$\text{margin} = \frac{1}{\lVert \mathbf{w} \rVert}.$$

最大化 $1/\lVert \mathbf{w} \rVert$ 等同於**最小化 $\lVert \mathbf{w} \rVert^2$**。支持向量機的目標函數正是由此而來。

### 硬間隔支持向量機

如果資料*確實*可以完美分隔，我們求解

$$\min_{\mathbf{w}, b} \ \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{subject to} \quad y_i (\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \ \ \forall i.$$

這個限制式表示每個點都在正確的一側*而且*位於街道之外。限制式恰好取等號（$y_i f(\mathbf{x}_i) = 1$）的那些點就是**支持向量**。

### 軟間隔 + 鉸鏈損失 + C

真實資料會重疊，所以我們透過**鬆弛變數 (slack)** $\xi_i \ge 0$ 允許違規，並為之付出代價：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \ \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i f(\mathbf{x}_i) \ge 1 - \xi_i, \ \ \xi_i \ge 0.$$

最佳的鬆弛變數恰好是 $\xi_i = \max(0,\, 1 - y_i f(\mathbf{x}_i))$，這就是**鉸鏈損失**。把它代回去，就把這個帶限制式的問題轉成了單純的正則化損失最小化：

$$\min_{\mathbf{w}, b} \ \underbrace{C \sum_i \max\big(0,\, 1 - y_i f(\mathbf{x}_i)\big)}_{\text{hinge loss: penalty for being inside/past the margin}} + \underbrace{\tfrac{1}{2}\lVert \mathbf{w} \rVert^2}_{\text{L2 reg = wide margin}}.$$

一旦一個點被正確分類*且留有餘裕*（$y_i f \ge 1$），鉸鏈損失就是零；當點悄悄越進街道或落到錯誤一側時，它會線性增長。對照第 04 課平滑的對數損失——鉸鏈損失在間隔為 1 處有一個硬「肘點」，並忽略那些已經穩穩分類正確的點。

- **$C$ = 正則化強度的倒數。** 大的 $C$ → 違規代價高昂 → 窄間隔、低偏差、高變異數（可能過度擬合）。小的 $C$ → 容忍違規 → 寬而平滑的間隔、較高偏差。$C$ 在這裡扮演的角色與脊迴歸中的 $1/\lambda$ 相同。

### 核技巧

這個問題的對偶形式（透過拉格朗日乘子 $\alpha_i \ge 0$ 導出）對資料的依賴**只透過內積** $\mathbf{x}_i^\top \mathbf{x}_j$：

$$\max_{\boldsymbol{\alpha}} \ \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j, \qquad \text{with } \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i.$$

只有支持向量的 $\alpha_i > 0$。現在見證魔法：把每一個內積 $\mathbf{x}_i^\top \mathbf{x}_j$ 替換成一個**核** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$，其中 $\phi$ 映射到某個高維空間。我們得到*在那個空間裡*的一個線性分隔器——也就是原始空間裡的一條彎曲邊界——而且我們從不需要顯式地計算 $\phi$。核不過就是兩點之間的一個**相似度分數**。

常見的核：

$$
\begin{aligned}
\text{Linear:} \quad & K(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top \mathbf{z} \\
\text{Polynomial (deg } p): \quad & K(\mathbf{x},\mathbf{z}) = (\gamma\,\mathbf{x}^\top \mathbf{z} + r)^{p} \\
\text{RBF / Gaussian:} \quad & K(\mathbf{x},\mathbf{z}) = \exp\!\big(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2\big)
\end{aligned}
$$

**RBF 核**是主力。把它讀作相似度：當兩點重合時它是 $1$，並隨著兩點分離而衰減趨向 $0$。**$\gamma$ 控制每個支持向量的影響範圍**：

- **大的 $\gamma$** → 快速衰減 → 每個支持向量只影響它極小的鄰域 → 彎彎曲曲、複雜的邊界（可能過度擬合）。
- **小的 $\gamma$** → 緩慢衰減 → 影響傳播得很遠 → 平滑、近乎線性的邊界（可能低度擬合）。

所以 $\gamma$ 控制邊界的*形狀 / 複雜度*，而 $C$ 控制*對誤差的容忍度*。你要把它們一起調校。

---

## 3. 程式碼

我們要分隔經典的**雙月牙 (two-moons)** 資料集——兩個交錯的新月形，沒有任何一條直線能把它們劈開——並觀察核如何改變一切。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. 建立一個無法線性分隔的資料集 ---------------------------
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 2. 支持向量機以距離為基礎，所以一定要先縮放特徵 ----------
# 用管線可保證縮放器只在訓練資料上擬合（避免資料洩漏）。
def make_svm(**kwargs):
    return make_pipeline(StandardScaler(), SVC(**kwargs))

models = {
    "Linear":        make_svm(kernel="linear", C=1.0),
    "Poly (deg 3)":  make_svm(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "RBF (gamma=1)": make_svm(kernel="rbf",  C=1.0, gamma=1.0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name:14s} test accuracy = {acc:.3f}")
# -> Linear         test accuracy = 0.892
# -> Poly (deg 3)   test accuracy = 0.867
# -> RBF (gamma=1)  test accuracy = 0.917
```

線性核之所以表現受限，是因為沒有任何一條線能劈開兩個月牙。RBF 核透過把邊界彎繞著新月形而勝出。注意三次多項式在這個雜訊水準下實際上表現得比線性*更差*——這是個有用的提醒：花俏的核不會自動變得更好；可靠的非線性預設是 RBF，而非多項式。

### 繪製決策邊界

```python
def plot_boundary(ax, model, X, y, title):
    # 建立一個涵蓋資料的網格，對每個網格點分類。
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")          # 填色區域
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
               edgecolors="k", s=18)                              # 資料點
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, model) in zip(axes, models.items()):
    plot_boundary(ax, model, X, y, name)
plt.tight_layout()
plt.show()
```

**你應該看到：** *Linear* 圖切出一條筆直的對角線，削掉了兩個月牙（把尖端誤分類了）。*Poly* 圖把邊界弓成一條溫和的近拋物線曲線，但仍無法完全包住新月形。*RBF* 圖則幾乎完美地貼合兩個月牙之間的縫隙——一條穿過空街的平滑 S 形曲線。

### 觀察 gamma 與 C 如何過度擬合

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
settings = [
    ("RBF gamma=0.1, C=1",  dict(gamma=0.1,  C=1)),
    ("RBF gamma=1,   C=1",  dict(gamma=1.0,  C=1)),
    ("RBF gamma=30,  C=1",  dict(gamma=30,   C=1)),    # 太彎曲 -> 過度擬合
    ("RBF gamma=1,   C=0.05", dict(gamma=1.0, C=0.05)),# 軟 -> 寬街
    ("RBF gamma=1,   C=1",  dict(gamma=1.0,  C=1)),
    ("RBF gamma=1,   C=100", dict(gamma=1.0, C=100)),  # 硬 -> 窄街
]
for ax, (title, kw) in zip(axes.ravel(), settings):
    m = make_svm(kernel="rbf", **kw).fit(X_train, y_train)
    plot_boundary(ax, m, X, y, f"{title}\n(test acc {m.score(X_test, y_test):.2f})")
plt.tight_layout()
plt.show()
```

**你應該看到：** 上排——當 $\gamma$ 從 0.1 攀升到 30，邊界從近乎筆直變形成緊緊纏繞著個別訓練點的、抖動的一團團小島（教科書級的過度擬合）。下排——當 $C$ 從 0.05 攀升到 100，間隔街道收窄，邊界扭曲變形以捕捉每一個最後的訓練點。

### 檢視支持向量

```python
svm = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(StandardScaler().fit_transform(X_train), y_train)
print("n_support_vectors per class:", svm.n_support_)
print("fraction of training set used:", svm.support_vectors_.shape[0] / len(X_train))
# -> n_support_vectors per class: [41 40]
# -> fraction of training set used: 0.289
```

只有約 29% 的點是支持向量；其餘的點對邊界毫不相干。正是這種稀疏性使得預測高效、模型穩健。

---

## 4. 實際案例 —— 小量標記資料上的聲學 / 振動故障偵測

在一艘無人水面載具 (USV) 上，你在推進軸附近裝一個加速度計或水下聽音器。你想在故障發生之前就標記出**空蝕 (cavitation)** 或**軸承磨損 (bearing wear)**。難處在於：你只負擔得起*標記*區區數十筆執行記錄（得有人去檢查硬體以確認真實標籤）。這正是支持向量機勝過隨機森林與神經網路的情境——**小量、乾淨、高維的資料**。

**管線。** 每段約 1 秒的振動片段變成一個特徵向量：一個 FFT 功率頻譜（或者一把 MFCC / 頻帶能量 + RMS + 峰度特徵）。這會給你，比方說，每個樣本 $d \approx 64$ 個特徵，但只有 $n \approx 80$ 段標記過的片段。當 $d$ 與 $n$ 相當時，深度網路會立刻過度擬合；而 RBF 支持向量機卻能茁壯成長，因為它的複雜度是由間隔控制的，而不是由參數數量。

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# X_spec: (80, 64) 頻帶能量頻譜, y: 0 = 健康, 1 = 故障
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
grid = GridSearchCV(
    pipe,
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]},
    cv=5, scoring="f1",          # 用 F1 是因為故障是稀有但重要的類別
)
# grid.fit(X_spec, y)
# print(grid.best_params_, grid.best_score_)
# -> {'svc__C': 10, 'svc__gamma': 0.1} 0.91
```

**為什麼核在這裡是正確的心智模型。** RBF 核衡量的是*頻譜相似度*：兩段頻率特徵相似的片段得到高 $K$，不相似的得到低 $K$。對一段新片段的判定，字面上就是一場加權投票——「這段跟故障的支持向量片段相比，跟健康的相比，有多相似？」——一名聲納操作員的直覺，被形式化了。`class_weight="balanced"` 處理了健康片段數量遠遠多於故障片段這件事。

同一套配方也可以套用到**聲納目標分類**（岩石 vs 金屬——UCI *Connectionist Bench Sonar* 資料集，$n=208$、$d=60$，一個支持向量機經典案例）、**光達回波材質分類**，或**慣性測量單元 (IMU) 步態 / 異常偵測**——任何具備豐富特徵卻標籤稀缺的情境。

---

## 5. 常見陷阱與技巧

- **永遠要縮放你的特徵。** 支持向量機衡量距離；一個未經縮放、數值範圍巨大的特徵（例如以 Pa 為單位的壓力 vs 一個正規化比值）會悄悄主宰整個核。把 `StandardScaler` 包進 `Pipeline`，讓縮放只在訓練折上擬合。
- **用交叉驗證一起調校 $C$ 與 $\gamma$。** 它們會交互作用——在某個 $\gamma$ 下好的 $C$，到另一個 $\gamma$ 下就錯了。在對數網格上使用 `GridSearchCV`（`C ∈ {0.1,1,10,100}`、`gamma ∈ {scale, 0.01, 0.1, 1}`）。預設的 `gamma="scale"` 是個合理的起點。
- **別去碰超過 3 次的多項式。** 高次多項式核在數值上不穩定，且容易發生狂野的外插。如果 RBF 做不到，多項式通常也做不到。
- **支持向量機在樣本數上擴展性很差。** 訓練在樣本數上大約是 $O(n^2)$ 到 $O(n^3)$，而且 `SVC` 會把支持向量保存在記憶體裡。超過約 5 萬到 10 萬列之後，就改用 `LinearSVC` / `SGDClassifier(loss="hinge")`，或轉向樹集成 / 神經網路。
- **原始的 `SVC` 分數不是機率。** `decision_function` 回傳的是有號間隔，而非校準過的機率。如果你需要機率來做門檻判定，請設定 `probability=True`（會加入 Platt 縮放，較慢）或用 `CalibratedClassifierCV` 包起來。
- **支持向量機 vs 樹，快速法則：** 對於小 / 中量資料、許多連續特徵、以及你預期決策面是平滑的情況（訊號、頻譜、嵌入），偏好**支持向量機**。對於大量表格資料、混合的類別 + 數值特徵、缺失值、以及你想要近乎零調整負擔的特徵重要性可解釋性時，偏好**樹集成**（第 06 課）。

---

## 6. 自我檢測

**Q1.** 兩條邊界都以 100% 準確率分隔你的訓練資料。為什麼支持向量機偏好間隔較大的那一條？

<details><summary>解答</summary>
較大的間隔對雜訊更穩健、泛化得更好：一個落點稍微偏離訓練資料的新點，由於有緩衝，仍會落在正確的一側。從幾何上看，最大化間隔等同於最小化 $\lVert\mathbf{w}\rVert$，也就是 L2 正則化——寬街邊界是「最簡單」的那一條，最不容易過度擬合。
</details>

**Q2.** 你刪掉所有訓練點，*只留下*支持向量，然後重新訓練。邊界會如何改變？

<details><summary>解答</summary>
完全不會改變。最佳的 $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ 對每一個非支持向量都有 $\alpha_i = 0$，所以只有支持向量會進入解之中。這正是支持向量機稀疏性與穩健性的來源。
</details>

**Q3.** 你的 RBF 支持向量機拿到 100% 訓練準確率，但只有 70% 測試準確率。你很可能把哪個旋鈕調得太高，又該往哪個方向轉？

<details><summary>解答</summary>
過度擬合——很可能是 $\gamma$ 太高（每個支持向量的影響太過局部，產生一條死記訓練集的彎曲邊界），以及／或 $C$ 太高（間隔太緊）。降低 $\gamma$ 並／或降低 $C$ 以平滑／加寬邊界，然後重新驗證。
</details>

**Q4.** 為什麼核被稱為「相似度函數」？對於兩個完全相同的點與兩個相距非常遠的點，RBF 核分別回傳什麼？

<details><summary>解答</summary>
$K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z})$ 是提升空間中的一個內積，而內積衡量的是對齊程度／相似度。對於 RBF，$K=\exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2)$：相同的點給出 $K=1$（最大相似度），相距遙遠的點給出 $K\to 0$。一次預測就是對支持向量做的一場相似度加權投票。
</details>

**Q5.** 你有 500 萬筆標記過、大多是類別型的表格遙測資料列。RBF `SVC` 是個好選擇嗎？你會改用什麼？

<details><summary>解答</summary>
不是。核 `SVC` 在樣本數上是 $O(n^2)$–$O(n^3)$，在數百萬列上會卡死，而且 RBF 無法優雅地處理類別型特徵。對於大量混合型別的表格資料，請用梯度提升樹集成（第 06 課）；如果你特別想要一個可大規模擴展的線性最大間隔模型，就用 `LinearSVC`／`SGDClassifier(loss="hinge")`。
</details>

---

## 回顧與下一步

- 支持向量機找出**最大間隔**超平面——類別之間最寬的空街——使其在構造上就具備穩健性。
- 只有**支持向量**（位於間隔上的點）定義了邊界；由 **$C$** 主導的軟間隔**鉸鏈損失 + L2** 目標函數，讓它能容忍重疊。
- **核技巧**用一個**相似度函數**替換內積，免費給你非線性邊界（線性 / 多項式 / RBF）；**$\gamma$** 設定 RBF 邊界的複雜度。
- 支持向量機在**小量、高維、連續**的資料（訊號、頻譜、聲納）上大放異彩；樹則在**大量、混合型別**的表格資料上勝出。

下一課我們將拋開標記資料，讓模型自行尋找結構：**[08 — 非監督式學習：k-平均與主成分分析](08-kmeans-pca.md)**。
