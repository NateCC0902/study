# 附錄 · 數學速查表

這是貫穿整個 22 課 ML/DL 課程所用公式的可快速瀏覽參考。每一條目都附上一行**內容/時機**說明，以及回到教學該公式那一課的連結。這是一張查詢表，而非教學文件——推導與直覺請見各課。

**符號說明：** 純量 $a$、向量 $\mathbf{x}$（行向量）、矩陣 $X$、$\odot$ = 逐元素（Hadamard）乘積、$\hat{y}$ = 預測、$m$ 或 $N$ = 樣本數、$\alpha$ = 學習率。

---

## 1. 線性代數要點

於 [`01-math-foundations.md`](01-math-foundations.md) 教學；張量形狀（shape）在各處反覆出現（尤其是 [`09-neural-networks-mlp.md`](09-neural-networks-mlp.md)）。

**內積（dot product）** — 兩向量的對齊程度／相似度；為零代表正交。

$$
\mathbf{a}\cdot\mathbf{b} = \mathbf{a}^\top\mathbf{b} = \sum_{i=1}^{n} a_i b_i = \lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert\cos\theta
$$

**餘弦相似度（cosine similarity）** — 僅看方向的相似度，與尺度無關；用於嵌入 ([`17-transfer-learning-llms-mlops.md`](17-transfer-learning-llms-mlops.md))。

$$
\cos\theta = \frac{\mathbf{a}^\top\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
$$

**矩陣–向量／矩陣–矩陣乘積** — 主力運算；內側維度必須一致。

$$
\underbrace{A}_{m\times n}\,\underbrace{\mathbf{x}}_{n\times 1} = \underbrace{\mathbf{b}}_{m\times 1},
\qquad
\underbrace{A}_{m\times n}\,\underbrace{B}_{n\times p} = \underbrace{C}_{m\times p},
\quad C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}
$$

**轉置、單位矩陣、反矩陣** — $I$ 是乘法的單位元素；$A^{-1}$ 還原 $A$（僅當方陣且非奇異時成立）。

$$
(AB)^\top = B^\top A^\top,\qquad AI = IA = A,\qquad A A^{-1} = A^{-1}A = I
$$

**L1 與 L2 範數** — 向量的「大小」；L2 是歐幾里得長度，L1 是絕對值之和（也是正則化的基礎，見 §7）。

$$
\lVert\mathbf{x}\rVert_1 = \sum_i |x_i|,
\qquad
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x}^\top\mathbf{x}}
$$

---

## 2. ML 所需的微積分

於 [`01-math-foundations.md`](01-math-foundations.md) 教學；它是訓練背後的引擎 ([`03-gradient-descent.md`](03-gradient-descent.md)、[`10-backpropagation.md`](10-backpropagation.md))。

**導數即斜率** — 局部的變化率；在極小值／極大值處為零。

$$
f'(x) = \frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

**梯度（gradient）** — 偏導數構成的向量；指向最陡的**上升**方向（所以我們沿 $-\nabla$ 下降）。

$$
\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots,\ \frac{\partial f}{\partial x_n}\right]^\top
$$

**連鎖律（chain rule）** — 沿著一條路徑組合各局部梯度；反向傳播 (backpropagation) 的種子。

$$
\frac{d}{dx}f\big(g(x)\big) = f'\big(g(x)\big)\,g'(x),
\qquad
\frac{\partial L}{\partial x} = \sum_{k}\frac{\partial L}{\partial u_k}\frac{\partial u_k}{\partial x}
$$

**中心有限差分** — 用於驗證反向傳播的數值梯度檢查 ([`10-backpropagation.md`](10-backpropagation.md))。

$$
\frac{\partial f}{\partial \theta} \approx \frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}
$$

---

## 3. 機率

於 [`01-math-foundations.md`](01-math-foundations.md) 教學；它支撐 MLE 損失（§4）與生成模型 ([`16-generative-models.md`](16-generative-models.md))。

**期望值（平均）與變異數** — 隨機變數的中心與離散程度。

$$
\mathbb{E}[X] = \mu = \sum_x x\,p(x)\ \ \text{(or }\textstyle\int x\,p(x)\,dx),
\qquad
\operatorname{Var}(X) = \mathbb{E}\big[(X-\mu)^2\big] = \mathbb{E}[X^2]-\mu^2
$$

**高斯（常態）分布** — 預設的雜訊／不確定性模型；平方誤差損失即其 MLE。

$$
\mathcal{N}(x\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**條件機率與貝氏定理** — 把 $p(\text{data}\mid\text{class})$ 反轉成 $p(\text{class}\mid\text{data})$；判別式／生成式之間的橋樑 ([`16-generative-models.md`](16-generative-models.md))。

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)},
\qquad
P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}
$$

**KL 散度** — 分布之間的距離；VAE ELBO 中的正則化項 ([`16-generative-models.md`](16-generative-models.md))。

$$
D_{\mathrm{KL}}(p\,\Vert\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$

---

## 4. 損失函數

迴歸損失見 [`02-linear-regression.md`](02-linear-regression.md)／[`05-overfitting-evaluation.md`](05-overfitting-evaluation.md)；分類損失見 [`04-logistic-regression.md`](04-logistic-regression.md)；鉸鏈損失見 [`07-svm-kernels.md`](07-svm-kernels.md)。簡潔的共用梯度 $X^\top(\hat{y}-y)$ 會在 [`10-backpropagation.md`](10-backpropagation.md) 再次出現。

**均方誤差（MSE）** — 迴歸的預設選擇；平滑、凸、對離群值敏感（高斯 MLE）。

$$
J_{\text{MSE}} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2,
\qquad
\nabla_{\mathbf{w}} J = \frac{2}{m}X^\top(X\mathbf{w}-\mathbf{y})
$$

> 本課程常會省略因子 2（被吸收進 $\alpha$）和／或改用 $\tfrac{1}{2m}$ 把它約掉：$\nabla J = \tfrac{1}{m}X^\top(X\mathbf{w}-\mathbf{y})$。

**平均絕對誤差（MAE）** — 對離群值穩健；梯度大小固定（在 0 處不平滑）。

$$
J_{\text{MAE}} = \frac{1}{m}\sum_{i=1}^{m}\lvert\hat{y}_i - y_i\rvert
$$

**二元交叉熵（BCE／log-loss）** — 兩類別分類；伯努利分布下的 MLE。搭配 sigmoid 輸出使用。

$$
J_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]
$$

Sigmoid + BCE 會得到約分後乾淨的梯度（沒有啟動函數的導數因子）：

$$
\nabla_{\mathbf{w}} J = \frac{1}{m}X^\top(\hat{\mathbf{y}}-\mathbf{y})
$$

**類別交叉熵（Categorical Cross-Entropy）** — 多類別；搭配 softmax。$y$ 為獨熱編碼，$k$ 為類別索引。

$$
J_{\text{CCE}} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log\hat{y}_{ik}
$$

Softmax + CCE 會塌縮成相同形式：$\partial J/\partial z_k = \hat{y}_k - y_k$。

**鉸鏈損失（軟間隔 SVM）** — 最大間隔分類，標籤 $y\in\{-1,+1\}$、分數 $z=\mathbf{w}^\top\mathbf{x}+b$；一旦某點被正確分類且超出間隔，損失即為零。

$$
J_{\text{hinge}} = \frac{1}{m}\sum_{i=1}^{m}\max\!\big(0,\ 1 - y_i z_i\big) + \frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2
$$

---

## 5. 梯度下降更新規則

原始梯度下降見 [`03-gradient-descent.md`](03-gradient-descent.md)；動量／RMSProp／Adam 見 [`12-training-deep-nets.md`](12-training-deep-nets.md)。它們都在最小化 $J(\theta)$。

**原始（批次）梯度下降** — 沿著梯度反方向走 $\alpha$ 倍的步長下坡。

$$
\theta \leftarrow \theta - \alpha\,\nabla_\theta J(\theta)
$$

> 對二次／MSE 碗狀曲面而言，穩定性要求 $\alpha < 2/\lambda_{\max}$（最大曲率）。小批次／SGD 在批次梯度 $\hat{g}$ 上沿用相同規則。

**動量（Momentum）** — 累積速度以滾過小凸起並抑制振盪。

$$
\mathbf{v} \leftarrow \beta\,\mathbf{v} + \nabla_\theta J,
\qquad
\theta \leftarrow \theta - \alpha\,\mathbf{v}
\qquad(\beta\approx 0.9)
$$

**Adam** — 逐參數的自適應學習率，結合動量（一階動差）+ RMSProp（二階動差），並加上偏差修正。

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^{\,t}},\qquad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \alpha\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
\end{aligned}
$$

> 預設值：$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$。**AdamW** 將權重衰減從梯度中解耦出來（見 §7）。

---

## 6. 啟動函數

於 [`09-neural-networks-mlp.md`](09-neural-networks-mlp.md) 教學；其導數驅動反向傳播 ([`10-backpropagation.md`](10-backpropagation.md)) 以及梯度消失的故事 ([`12-training-deep-nets.md`](12-training-deep-nets.md))。

**Sigmoid** — 壓縮到 $(0,1)$；機率／閘門輸出。會飽和 → 梯度消失。

$$
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)
$$

**Tanh** — sigmoid 的零中心化 $(-1,1)$ 版本；同樣會飽和。

$$
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},
\qquad
\tanh'(z) = 1 - \tanh^2(z)
$$

**ReLU** — 預設的隱藏層啟動函數；廉價、在 $z>0$ 時不飽和，修正梯度消失。

$$
\operatorname{ReLU}(z) = \max(0, z),
\qquad
\operatorname{ReLU}'(z) = \begin{cases}1 & z>0\\ 0 & z<0\end{cases}
$$

**Softmax** — 把一個 logits 向量轉成機率分布；多類別輸出（搭配 CCE）。

$$
\operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

> 雅可比矩陣：$\partial\,\text{softmax}_i/\partial z_j = \hat{y}_i(\delta_{ij}-\hat{y}_j)$。搭配 CCE 時會簡化為 $\hat{y}_k - y_k$。

---

## 7. 正則化項

於 [`05-overfitting-evaluation.md`](05-overfitting-evaluation.md) 教學；權重衰減在 [`12-training-deep-nets.md`](12-training-deep-nets.md) 再次探討。加進損失裡以懲罰過大的權重並抑制過度擬合。

**L2（Ridge／權重衰減）** — 把所有權重平滑地收縮向零；可微分。

$$
J_{\text{total}} = J_{\text{data}} + \frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2,
\qquad
\nabla\!\left(\tfrac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2\right) = \lambda\mathbf{w}
$$

> $\lambda\mathbf{w}$ 這個梯度就是為什麼 L2 ≡「權重衰減」：每一步也會把 $\mathbf{w}$ 乘上 $(1-\alpha\lambda)$。**AdamW** 直接套用這個衰減，而不是透過梯度。

**L1（Lasso）** — 把權重精確逼到零 → 稀疏的特徵選擇；在 0 處不平滑。

$$
J_{\text{total}} = J_{\text{data}} + \lambda\lVert\mathbf{w}\rVert_1,
\qquad
\nabla\!\big(\lambda\lVert\mathbf{w}\rVert_1\big) = \lambda\,\operatorname{sign}(\mathbf{w})
$$

---

## 8. 關鍵指標

於 [`05-overfitting-evaluation.md`](05-overfitting-evaluation.md) 教學。源自混淆矩陣：TP、FP、FN、TN。當準確率在不平衡資料上會誤導時使用。

**精確率（Precision）** — 在預測為正的當中，有多少是正確的（懲罰假警報）。

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**召回率（Recall，敏感度／TPR）** — 在實際為正的當中，有多少被抓到（懲罰漏失）。

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1 分數** — 平衡精確率與召回率的調和平均。

$$
F_1 = \frac{2\,\cdot\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}
$$

**準確率（Accuracy）** — 整體答對的比例；在不平衡資料上會造成誤導。

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**R²（決定係數）** — 迴歸：所解釋的變異數比例；$1$ 為完美，$0$ = 預測平均值。

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

**RMSE** — 以目標自身單位表示的迴歸誤差。

$$
\text{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2}
$$

---

## 9. 注意力

於 [`15-attention-transformers.md`](15-attention-transformers.md) 教學。Transformer 與大型語言模型的核心運算。

**縮放點積注意力** — 一種柔性、可微分的資料庫查詢：查詢 $Q$ 對鍵 $K$ 評分，softmax 權重再取出值 $V$ 的混合。$\sqrt{d_k}$ 的縮放可避免 softmax 梯度在大維度時消失。

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

**多頭注意力** — 在 $h$ 個平行的子空間中各跑一次注意力，然後串接再投影。

$$
\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,
\quad
\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**正弦位置編碼** — 注入順序資訊，因為注意力對排列順序是盲的。會加到詞元嵌入上。

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),
\qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

> **因果遮罩（causal mask）：** 在 softmax 之前把未來位置的分數設為 $-\infty$，使解碼器詞元無法看到前方。代價是在序列長度 $n$ 上為 $O(n^2)$，換來完全的平行化與固定長度的長程路徑。

---

## 10. 強化學習

於 [`18-reinforcement-learning.md`](18-reinforcement-learning.md) 教學。代理人透過最大化期望折扣獎勵來學習策略——這正是載具控制背後的範式。

**折扣報酬** — 代理人所要最大化的目標；$\gamma\in[0,1)$ 在即時與未來獎勵間取捨，$r_t$ = 第 $t$ 步的獎勵。

$$
G_t = \sum_{k=0}^{\infty}\gamma^{k}\, r_{t+k+1}
$$

**貝爾曼最佳性（動作價值）** — 最佳的 $Q^\star(s,a)$ 是自我一致的：先採取 $a$，之後永遠採取貪婪動作。

$$
Q^\star(s,a) = \mathbb{E}\big[\, r + \gamma \max_{a'} Q^\star(s',a') \,\big]
$$

**Q 學習更新** — 朝貝爾曼目標前進的離策略時間差分步驟（$\alpha$ = 學習率，$s'$ = 下一個狀態）。

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]
$$

> 行為採 **ε-貪婪**：以機率 $\epsilon$ 採取隨機動作（探索），否則取 $\arg\max_a Q(s,a)$（利用）。**DQN** 把這張表換成網路 + 回放緩衝區 + 目標網路。

**策略梯度（REINFORCE）** — 直接拉高各動作的對數機率，並以該動作所獲得的報酬加權。

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\, \nabla_\theta \log \pi_\theta(a\mid s)\, G_t \,\big]
$$

---

## 11. 偵測與分割指標

於 [`19-detection-segmentation.md`](19-detection-segmentation.md) 教學。重點是定位品質，而不只是類別標籤。

**交集比聯集（IoU／Jaccard）** — 框／遮罩的重疊程度；當與真實標註的 IoU $\geq$ 某門檻（常用 0.5）時，該次偵測才算正確。

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

**平均精確度／mAP** — AP 是單一類別之精確率–召回率曲線下的面積；**mAP** 則在全部 $C$ 個類別上對 AP 取平均（且通常還會跨多個 IoU 門檻取平均）。

$$
\text{AP} = \int_0^1 p(r)\,dr, \qquad \text{mAP} = \frac{1}{C}\sum_{c=1}^{C}\text{AP}_c
$$

> **NMS**（非極大值抑制）：保留分數最高的框，丟棄與其 IoU 高於門檻的重疊框——移除對同一物件的重複偵測。

---

## 12. 資料與模型選擇

資料準備見 [`20-data-feature-engineering.md`](20-data-feature-engineering.md)；調參見 [`21-hyperparameter-optimization.md`](21-hyperparameter-optimization.md)。

**平衡類別權重** — 提高稀有類別的權重，讓損失不再忽略它們（$N$ 個樣本、$K$ 個類別、類別 $c$ 中有 $n_c$ 個）。

$$
w_c = \frac{N}{K\,n_c}
$$

**隨機搜尋的命中覆蓋率** — $n$ 次隨機試驗落入最佳 $f$ 比例設定的機率；這就是為什麼約 60 次試驗勝過粗糙的網格（例如 $f=0.05,\ n=60 \Rightarrow \approx 95\%$）。

$$
P(\text{hit}) = 1 - (1-f)^{n}
$$

> **巢狀交叉驗證：** 內層迴圈選擇超參數，外層迴圈估計效能——這是同時調參*又*回報結果唯一誠實的做法。若在測試集上調參，你發表的每個數字都會帶有樂觀偏差。
