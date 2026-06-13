# Appendix · Math Cheatsheet

A scannable quick-reference for the formulas used throughout the 22-lesson ML/DL course. Each entry has a one-line **what/when** note and a link back to the lesson where it's taught. This is a lookup table, not a tutorial — see the lessons for derivations and intuition.

**Notation:** scalars $a$, vectors $\mathbf{x}$ (column), matrices $X$, $\odot$ = elementwise (Hadamard) product, $\hat{y}$ = prediction, $m$ or $N$ = number of examples, $\alpha$ = learning rate.

---

## 1. Linear Algebra Essentials

Taught in [`01-math-foundations.md`](01-math-foundations.md); shapes recur everywhere (especially [`09-neural-networks-mlp.md`](09-neural-networks-mlp.md)).

**Dot product** — alignment / similarity of two vectors; zero ⇒ orthogonal.

$$
\mathbf{a}\cdot\mathbf{b} = \mathbf{a}^\top\mathbf{b} = \sum_{i=1}^{n} a_i b_i = \lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert\cos\theta
$$

**Cosine similarity** — direction-only similarity, scale-invariant; used for embeddings ([`17-transfer-learning-llms-mlops.md`](17-transfer-learning-llms-mlops.md)).

$$
\cos\theta = \frac{\mathbf{a}^\top\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
$$

**Matrix–vector / matrix–matrix product** — the workhorse; inner dimensions must match.

$$
\underbrace{A}_{m\times n}\,\underbrace{\mathbf{x}}_{n\times 1} = \underbrace{\mathbf{b}}_{m\times 1},
\qquad
\underbrace{A}_{m\times n}\,\underbrace{B}_{n\times p} = \underbrace{C}_{m\times p},
\quad C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}
$$

**Transpose, identity, inverse** — $I$ is the multiplicative identity; $A^{-1}$ undoes $A$ (only if square and non-singular).

$$
(AB)^\top = B^\top A^\top,\qquad AI = IA = A,\qquad A A^{-1} = A^{-1}A = I
$$

**L1 and L2 norms** — vector "size"; L2 is Euclidean length, L1 is sum of absolute values (and the basis of regularization, §7).

$$
\lVert\mathbf{x}\rVert_1 = \sum_i |x_i|,
\qquad
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x}^\top\mathbf{x}}
$$

---

## 2. Calculus for ML

Taught in [`01-math-foundations.md`](01-math-foundations.md); the engine behind training ([`03-gradient-descent.md`](03-gradient-descent.md), [`10-backpropagation.md`](10-backpropagation.md)).

**Derivative as slope** — local rate of change; zero at minima/maxima.

$$
f'(x) = \frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

**Gradient** — vector of partials; points in the direction of steepest **ascent** (so we descend along $-\nabla$).

$$
\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots,\ \frac{\partial f}{\partial x_n}\right]^\top
$$

**Chain rule** — compose local gradients along a path; the seed of backpropagation.

$$
\frac{d}{dx}f\big(g(x)\big) = f'\big(g(x)\big)\,g'(x),
\qquad
\frac{\partial L}{\partial x} = \sum_{k}\frac{\partial L}{\partial u_k}\frac{\partial u_k}{\partial x}
$$

**Centered finite difference** — numerical gradient check for verifying backprop ([`10-backpropagation.md`](10-backpropagation.md)).

$$
\frac{\partial f}{\partial \theta} \approx \frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}
$$

---

## 3. Probability

Taught in [`01-math-foundations.md`](01-math-foundations.md); underpins MLE losses (§4) and generative models ([`16-generative-models.md`](16-generative-models.md)).

**Expectation (mean) and variance** — center and spread of a random variable.

$$
\mathbb{E}[X] = \mu = \sum_x x\,p(x)\ \ \text{(or }\textstyle\int x\,p(x)\,dx),
\qquad
\operatorname{Var}(X) = \mathbb{E}\big[(X-\mu)^2\big] = \mathbb{E}[X^2]-\mu^2
$$

**Gaussian (normal) distribution** — the default noise/uncertainty model; squared-error loss is its MLE.

$$
\mathcal{N}(x\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Conditional probability & Bayes' rule** — invert $p(\text{data}\mid\text{class})$ into $p(\text{class}\mid\text{data})$; the discriminative/generative link ([`16-generative-models.md`](16-generative-models.md)).

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)},
\qquad
P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}
$$

**KL divergence** — distance between distributions; the regularizer term in the VAE ELBO ([`16-generative-models.md`](16-generative-models.md)).

$$
D_{\mathrm{KL}}(p\,\Vert\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$

---

## 4. Loss Functions

Regression losses in [`02-linear-regression.md`](02-linear-regression.md)/[`05-overfitting-evaluation.md`](05-overfitting-evaluation.md); classification losses in [`04-logistic-regression.md`](04-logistic-regression.md); hinge in [`07-svm-kernels.md`](07-svm-kernels.md). The clean shared gradient $X^\top(\hat{y}-y)$ recurs in [`10-backpropagation.md`](10-backpropagation.md).

**Mean Squared Error (MSE)** — regression default; smooth, convex, outlier-sensitive (Gaussian MLE).

$$
J_{\text{MSE}} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2,
\qquad
\nabla_{\mathbf{w}} J = \frac{2}{m}X^\top(X\mathbf{w}-\mathbf{y})
$$

> The course often drops the factor 2 (absorbed into $\alpha$) and/or uses $\tfrac{1}{2m}$ to cancel it: $\nabla J = \tfrac{1}{m}X^\top(X\mathbf{w}-\mathbf{y})$.

**Mean Absolute Error (MAE)** — robust to outliers; constant-magnitude gradient (non-smooth at 0).

$$
J_{\text{MAE}} = \frac{1}{m}\sum_{i=1}^{m}\lvert\hat{y}_i - y_i\rvert
$$

**Binary Cross-Entropy (BCE / log-loss)** — two-class classification; MLE under Bernoulli. Use with sigmoid output.

$$
J_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]
$$

Sigmoid + BCE gives the clean cancelled gradient (no activation-derivative factor):

$$
\nabla_{\mathbf{w}} J = \frac{1}{m}X^\top(\hat{\mathbf{y}}-\mathbf{y})
$$

**Categorical Cross-Entropy** — multi-class; pair with softmax. $y$ is one-hot, $k$ indexes classes.

$$
J_{\text{CCE}} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log\hat{y}_{ik}
$$

Softmax + CCE collapses to the same form: $\partial J/\partial z_k = \hat{y}_k - y_k$.

**Hinge loss (soft-margin SVM)** — max-margin classification with labels $y\in\{-1,+1\}$ and score $z=\mathbf{w}^\top\mathbf{x}+b$; zero loss once a point is correctly classified beyond the margin.

$$
J_{\text{hinge}} = \frac{1}{m}\sum_{i=1}^{m}\max\!\big(0,\ 1 - y_i z_i\big) + \frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2
$$

---

## 5. Gradient Descent Update Rules

Vanilla GD in [`03-gradient-descent.md`](03-gradient-descent.md); momentum / RMSProp / Adam in [`12-training-deep-nets.md`](12-training-deep-nets.md). All minimize $J(\theta)$.

**Vanilla (batch) gradient descent** — step downhill by $\alpha$ times the gradient.

$$
\theta \leftarrow \theta - \alpha\,\nabla_\theta J(\theta)
$$

> Stability for quadratic/MSE bowls requires $\alpha < 2/\lambda_{\max}$ (largest curvature). Mini-batch/SGD uses the same rule on a batch gradient $\hat{g}$.

**Momentum** — accumulate a velocity to roll through small bumps and dampen oscillation.

$$
\mathbf{v} \leftarrow \beta\,\mathbf{v} + \nabla_\theta J,
\qquad
\theta \leftarrow \theta - \alpha\,\mathbf{v}
\qquad(\beta\approx 0.9)
$$

**Adam** — per-parameter adaptive LR combining momentum (1st moment) + RMSProp (2nd moment), with bias correction.

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^{\,t}},\qquad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \alpha\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
\end{aligned}
$$

> Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. **AdamW** decouples weight decay from the gradient (see §7).

---

## 6. Activation Functions

Taught in [`09-neural-networks-mlp.md`](09-neural-networks-mlp.md); derivatives drive backprop ([`10-backpropagation.md`](10-backpropagation.md)) and the vanishing-gradient story ([`12-training-deep-nets.md`](12-training-deep-nets.md)).

**Sigmoid** — squashes to $(0,1)$; probability/gate output. Saturates → vanishing gradients.

$$
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)
$$

**Tanh** — zero-centered $(-1,1)$ version of sigmoid; also saturates.

$$
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},
\qquad
\tanh'(z) = 1 - \tanh^2(z)
$$

**ReLU** — default hidden activation; cheap, non-saturating for $z>0$, fixes vanishing gradients.

$$
\operatorname{ReLU}(z) = \max(0, z),
\qquad
\operatorname{ReLU}'(z) = \begin{cases}1 & z>0\\ 0 & z<0\end{cases}
$$

**Softmax** — turns a vector of logits into a probability distribution; multi-class output (pairs with CCE).

$$
\operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

> Jacobian: $\partial\,\text{softmax}_i/\partial z_j = \hat{y}_i(\delta_{ij}-\hat{y}_j)$. With CCE it simplifies to $\hat{y}_k - y_k$.

---

## 7. Regularization Terms

Taught in [`05-overfitting-evaluation.md`](05-overfitting-evaluation.md); weight decay revisited in [`12-training-deep-nets.md`](12-training-deep-nets.md). Added to the loss to penalize large weights and curb overfitting.

**L2 (Ridge / weight decay)** — shrinks all weights smoothly toward zero; differentiable.

$$
J_{\text{total}} = J_{\text{data}} + \frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2,
\qquad
\nabla\!\left(\tfrac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2\right) = \lambda\mathbf{w}
$$

> The $\lambda\mathbf{w}$ gradient is why L2 ≡ "weight decay": each step also multiplies $\mathbf{w}$ by $(1-\alpha\lambda)$. **AdamW** applies this decay directly rather than through the gradient.

**L1 (Lasso)** — drives weights exactly to zero → sparse feature selection; non-smooth at 0.

$$
J_{\text{total}} = J_{\text{data}} + \lambda\lVert\mathbf{w}\rVert_1,
\qquad
\nabla\!\big(\lambda\lVert\mathbf{w}\rVert_1\big) = \lambda\,\operatorname{sign}(\mathbf{w})
$$

---

## 8. Key Metrics

Taught in [`05-overfitting-evaluation.md`](05-overfitting-evaluation.md). From the confusion matrix: TP, FP, FN, TN. Use when accuracy lies on imbalanced data.

**Precision** — of predicted positives, how many are correct (penalizes false alarms).

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall (sensitivity / TPR)** — of actual positives, how many were caught (penalizes misses).

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1 score** — harmonic mean balancing precision and recall.

$$
F_1 = \frac{2\,\cdot\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Accuracy** — overall fraction correct; misleading on imbalanced data.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**R² (coefficient of determination)** — regression: fraction of variance explained; $1$ is perfect, $0$ = predicting the mean.

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

**RMSE** — regression error in the target's own units.

$$
\text{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2}
$$

---

## 9. Attention

Taught in [`15-attention-transformers.md`](15-attention-transformers.md). The core operation of Transformers and LLMs.

**Scaled dot-product attention** — a soft, differentiable database lookup: queries $Q$ score against keys $K$, the softmax weights retrieve a blend of values $V$. The $\sqrt{d_k}$ scaling keeps softmax gradients from vanishing at large dimension.

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

**Multi-head attention** — run $h$ attentions in parallel subspaces, then concatenate and project.

$$
\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,
\quad
\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Sinusoidal positional encoding** — injects order, since attention is permutation-blind. Added to token embeddings.

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),
\qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

> **Causal mask:** set scores for future positions to $-\infty$ before softmax so a decoder token can't see ahead. Cost is $O(n^2)$ in sequence length $n$, traded for full parallelism and constant-length long-range paths.

---

## 10. Reinforcement Learning

Taught in [`18-reinforcement-learning.md`](18-reinforcement-learning.md). The agent learns a policy by maximizing expected discounted reward — the paradigm behind vehicle control.

**Discounted return** — what the agent maximizes; $\gamma\in[0,1)$ trades immediate vs future reward, $r_t$ = reward at step $t$.

$$
G_t = \sum_{k=0}^{\infty}\gamma^{k}\, r_{t+k+1}
$$

**Bellman optimality (action-value)** — the optimal $Q^\star(s,a)$ is self-consistent: take $a$, then act greedily forever after.

$$
Q^\star(s,a) = \mathbb{E}\big[\, r + \gamma \max_{a'} Q^\star(s',a') \,\big]
$$

**Q-learning update** — off-policy temporal-difference step toward the Bellman target ($\alpha$ = learning rate, $s'$ = next state).

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]
$$

> Behavior is **ε-greedy**: random action w.p. $\epsilon$ (explore), else $\arg\max_a Q(s,a)$ (exploit). **DQN** swaps the table for a network + replay buffer + target net.

**Policy gradient (REINFORCE)** — directly push up the log-probability of actions weighted by the return they earned.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\, \nabla_\theta \log \pi_\theta(a\mid s)\, G_t \,\big]
$$

---

## 11. Detection & Segmentation Metrics

Taught in [`19-detection-segmentation.md`](19-detection-segmentation.md). Localization quality, not just the class label.

**Intersection over Union (IoU / Jaccard)** — box/mask overlap; a detection counts as correct when IoU with the ground truth $\geq$ a threshold (commonly 0.5).

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Average Precision / mAP** — AP is the area under the precision–recall curve for one class; **mAP** averages AP over all $C$ classes (and often over IoU thresholds).

$$
\text{AP} = \int_0^1 p(r)\,dr, \qquad \text{mAP} = \frac{1}{C}\sum_{c=1}^{C}\text{AP}_c
$$

> **NMS** (non-max suppression): keep the highest-scoring box, drop overlapping boxes with IoU above a threshold — removes duplicate detections of the same object.

---

## 12. Data & Model Selection

Data prep in [`20-data-feature-engineering.md`](20-data-feature-engineering.md); tuning in [`21-hyperparameter-optimization.md`](21-hyperparameter-optimization.md).

**Balanced class weight** — up-weight rare classes so the loss stops ignoring them ($N$ samples, $K$ classes, $n_c$ in class $c$).

$$
w_c = \frac{N}{K\,n_c}
$$

**Random-search coverage** — probability that $n$ random trials land in the best fraction $f$ of configs; why ~60 trials beats a coarse grid (e.g. $f=0.05,\ n=60 \Rightarrow \approx 95\%$).

$$
P(\text{hit}) = 1 - (1-f)^{n}
$$

> **Nested CV:** the inner loop selects hyperparameters, the outer loop estimates performance — the only honest way to tune *and* report. Tune on your test set and every number you publish is optimistically biased.
