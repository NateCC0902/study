# 07 — Support Vector Machines & Kernels

> Part 2 · Lesson 07 · Code stack: scikit-learn

**Prerequisites:** [06 — k-NN, Decision Trees & Ensembles](06-knn-trees-ensembles.md)

**By the end you can:**
- Explain *why* the **maximum-margin** hyperplane is the "safest" linear separator, and identify the **support vectors** that define it.
- Connect the SVM objective to the **hinge loss** + L2 regularization, and tune the **soft-margin C** parameter.
- Use the **kernel trick** to separate non-linear data without ever computing the high-dimensional features, picking sensibly between **linear / polynomial / RBF** kernels and tuning **gamma**.
- Train and visualize `SVC` decision boundaries in scikit-learn, and decide when an SVM beats a tree ensemble.

---

## 1. Intuition

A logistic-regression boundary (Lesson 04) is happy as long as points land on the correct side. Two different boundaries that both separate the data perfectly look equally good to it. But they are *not* equally good — one might pass a hair's breadth from your nearest data points, the other might split the gap right down the middle.

A **Support Vector Machine (SVM)** picks the line that sits in the *middle of the widest empty street* between the two classes. Think of bulldozing the fattest possible road through the gap between the clusters: the **margin** is the width of that road, and the boundary is its center line. The handful of points touching the curbs are the **support vectors** — they alone determine the road. Every other point could be deleted and the boundary would not move.

```mermaid
flowchart LR
    A["Many separating lines<br/>fit the data"] --> B["SVM rule:<br/>maximize the margin<br/>(widest street)"]
    B --> C["Boundary defined only by<br/>support vectors on the curb"]
    C --> D["Data not linearly separable?"]
    D -->|"lift to higher dim<br/>via a kernel"| E["Linear street there<br/>= curvy boundary here"]
```

Why is the widest street the smart choice? It is the most *robust* to noise. A new measurement that lands slightly off from training points still falls on the correct side, because you left the maximum buffer. This is a built-in form of regularization — large margin is the geometric cousin of "keep the model simple" from Lesson 05.

The second big idea: real sensor data is rarely separable by a straight line. The **kernel trick** lets the SVM behave as if it lifted your points into a much higher-dimensional space where a flat plane *does* separate them — without ever paying the cost of computing those coordinates. A curved boundary in your 2D world is just a straight cut in that hidden high-D world.

---

## 2. The Math

### The separating hyperplane

A linear classifier is a hyperplane

$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$$

- $\mathbf{x} \in \mathbb{R}^d$ — a feature vector (e.g. $d$ sensor readings).
- $\mathbf{w} \in \mathbb{R}^d$ — the weight vector, **normal (perpendicular) to the boundary**.
- $b \in \mathbb{R}$ — the bias / offset that shifts the plane away from the origin.

We predict class $\hat{y} = \operatorname{sign}(f(\mathbf{x}))$, with labels coded as $y \in \{-1, +1\}$ (note: SVMs use $\pm 1$, not $0/1$).

### Margin = distance to the closest point

The signed distance from a point $\mathbf{x}$ to the hyperplane is $f(\mathbf{x}) / \lVert \mathbf{w} \rVert$ (standard point-to-plane formula from Lesson 01). We are free to rescale $\mathbf{w}, b$ however we like, so we fix the scale by demanding that the closest points satisfy $|f(\mathbf{x})| = 1$. Those points sit on the **margin boundaries** $f(\mathbf{x}) = \pm 1$, and the street's half-width is

$$\text{margin} = \frac{1}{\lVert \mathbf{w} \rVert}.$$

Maximizing $1/\lVert \mathbf{w} \rVert$ is the same as **minimizing $\lVert \mathbf{w} \rVert^2$**. That is where the SVM objective comes from.

### Hard-margin SVM

If the data *is* perfectly separable, we solve

$$\min_{\mathbf{w}, b} \ \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{subject to} \quad y_i (\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \ \ \forall i.$$

The constraint says every point is on the correct side *and* outside the street. The points where the constraint is tight ($y_i f(\mathbf{x}_i) = 1$) are the **support vectors**.

### Soft margin + the hinge loss + C

Real data overlaps, so we allow violations via **slack** $\xi_i \ge 0$ and pay for them:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \ \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i f(\mathbf{x}_i) \ge 1 - \xi_i, \ \ \xi_i \ge 0.$$

The optimal slack is exactly $\xi_i = \max(0,\, 1 - y_i f(\mathbf{x}_i))$, which is the **hinge loss**. Substituting it in turns the constrained problem into plain regularized loss minimization:

$$\min_{\mathbf{w}, b} \ \underbrace{C \sum_i \max\big(0,\, 1 - y_i f(\mathbf{x}_i)\big)}_{\text{hinge loss: penalty for being inside/past the margin}} + \underbrace{\tfrac{1}{2}\lVert \mathbf{w} \rVert^2}_{\text{L2 reg = wide margin}}.$$

The hinge loss is zero once a point is correctly classified *with room to spare* ($y_i f \ge 1$); it grows linearly as the point creeps into the street or onto the wrong side. Compare to the smooth log-loss of Lesson 04 — the hinge has a hard "elbow" at margin 1 and ignores points that are already comfortably correct.

- **$C$ = inverse regularization strength.** Large $C$ → violations are expensive → narrow margin, low bias, high variance (can overfit). Small $C$ → tolerate violations → wide, smooth margin, more bias. $C$ plays the same role here as $1/\lambda$ in ridge.

### The kernel trick

The dual form of the problem (derived via Lagrange multipliers $\alpha_i \ge 0$) depends on the data **only through inner products** $\mathbf{x}_i^\top \mathbf{x}_j$:

$$\max_{\boldsymbol{\alpha}} \ \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j, \qquad \text{with } \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i.$$

Only support vectors have $\alpha_i > 0$. Now the magic: replace every inner product $\mathbf{x}_i^\top \mathbf{x}_j$ with a **kernel** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$, where $\phi$ maps to some high-dimensional space. We get a linear separator *in that space* — a curved boundary in the original space — and we never compute $\phi$ explicitly. A kernel is just a **similarity score** between two points.

Common kernels:

$$
\begin{aligned}
\text{Linear:} \quad & K(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top \mathbf{z} \\
\text{Polynomial (deg } p): \quad & K(\mathbf{x},\mathbf{z}) = (\gamma\,\mathbf{x}^\top \mathbf{z} + r)^{p} \\
\text{RBF / Gaussian:} \quad & K(\mathbf{x},\mathbf{z}) = \exp\!\big(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2\big)
\end{aligned}
$$

The **RBF kernel** is the workhorse. Read it as similarity: it is $1$ when two points coincide and decays toward $0$ as they separate. **$\gamma$ controls the reach** of each support vector:

- **Large $\gamma$** → fast decay → each support vector influences only its tiny neighborhood → wiggly, complex boundary (can overfit).
- **Small $\gamma$** → slow decay → influence spreads far → smooth, almost-linear boundary (can underfit).

So $\gamma$ controls boundary *shape/complexity* while $C$ controls *tolerance for errors*. You tune them together.

---

## 3. Code

We will separate the classic **two-moons** dataset — two interlocking crescents that no straight line can split — and watch how the kernel changes everything.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Build a non-linearly-separable dataset ---------------------------
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 2. SVMs are distance-based, so ALWAYS scale features first ----------
# A pipeline guarantees the scaler is fit on train data only (no leakage).
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

The linear kernel is held back because no line splits the moons. The RBF kernel wins by bending the boundary around the crescents. Note the degree-3 polynomial actually does *worse* than linear here at this noise level — a useful reminder that a fancier kernel is not automatically better; RBF, not poly, is the reliable non-linear default.

### Plotting the decision boundaries

```python
def plot_boundary(ax, model, X, y, title):
    # Build a grid covering the data, classify every grid point.
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")          # filled regions
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
               edgecolors="k", s=18)                              # data points
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, model) in zip(axes, models.items()):
    plot_boundary(ax, model, X, y, name)
plt.tight_layout()
plt.show()
```

**What you should SEE:** the *Linear* panel cuts a straight diagonal that clips both moons (misclassifying the tips). The *Poly* panel bows the boundary into a gentle parabola-like curve that still can't fully wrap the crescents. The *RBF* panel hugs the gap between the two moons almost perfectly — a smooth S-curve threading the empty street.

### Watching gamma and C overfit

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
settings = [
    ("RBF gamma=0.1, C=1",  dict(gamma=0.1,  C=1)),
    ("RBF gamma=1,   C=1",  dict(gamma=1.0,  C=1)),
    ("RBF gamma=30,  C=1",  dict(gamma=30,   C=1)),    # too wiggly -> overfit
    ("RBF gamma=1,   C=0.05", dict(gamma=1.0, C=0.05)),# soft -> wide street
    ("RBF gamma=1,   C=1",  dict(gamma=1.0,  C=1)),
    ("RBF gamma=1,   C=100", dict(gamma=1.0, C=100)),  # hard -> tight street
]
for ax, (title, kw) in zip(axes.ravel(), settings):
    m = make_svm(kernel="rbf", **kw).fit(X_train, y_train)
    plot_boundary(ax, m, X, y, f"{title}\n(test acc {m.score(X_test, y_test):.2f})")
plt.tight_layout()
plt.show()
```

**What you should SEE:** top row — as $\gamma$ climbs from 0.1 → 30 the boundary morphs from nearly straight to a jittery blob of islands wrapped tightly around individual training points (textbook overfit). Bottom row — as $C$ climbs from 0.05 → 100 the margin street narrows and the boundary contorts to catch every last training point.

### Inspecting support vectors

```python
svm = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(StandardScaler().fit_transform(X_train), y_train)
print("n_support_vectors per class:", svm.n_support_)
print("fraction of training set used:", svm.support_vectors_.shape[0] / len(X_train))
# -> n_support_vectors per class: [41 40]
# -> fraction of training set used: 0.289
```

Only ~29% of points are support vectors; the rest are irrelevant to the boundary. That sparsity is what makes prediction efficient and the model robust.

---

## 4. Real Case — Acoustic / vibration fault detection on small labeled data

On a USV you mount an accelerometer or hydrophone near the propulsion shaft. You want to flag **cavitation** or **bearing wear** before it becomes a failure. The catch: you can only afford to *label* a few dozen runs (someone has to inspect the hardware to confirm ground truth). This is exactly the regime where SVMs beat random forests and neural nets — **small, clean, high-dimensional data**.

**Pipeline.** Each ~1-second vibration clip becomes a feature vector: an FFT power spectrum (or a handful of MFCC / band-energy + RMS + kurtosis features). That gives you, say, $d \approx 64$ features per sample but only $n \approx 80$ labeled clips. With $d$ comparable to $n$, a deep net would overfit instantly; an RBF SVM thrives because its complexity is controlled by the margin, not by parameter count.

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# X_spec: (80, 64) band-energy spectra, y: 0 = healthy, 1 = faulty
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
grid = GridSearchCV(
    pipe,
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]},
    cv=5, scoring="f1",          # F1 because faults are the rare, important class
)
# grid.fit(X_spec, y)
# print(grid.best_params_, grid.best_score_)
# -> {'svc__C': 10, 'svc__gamma': 0.1} 0.91
```

**Why the kernel is the right mental model here.** The RBF kernel measures *spectral similarity*: two clips with similar frequency signatures get a high $K$, dissimilar ones get a low $K$. The decision for a new clip is literally a weighted vote of "how similar is this to the faulty support-vector clips vs the healthy ones?" — a sonar-operator's intuition, formalized. `class_weight="balanced"` handles the fact that healthy clips vastly outnumber faulty ones.

The same recipe maps onto **sonar target classification** (rock vs metal — the UCI *Connectionist Bench Sonar* dataset, $n=208$, $d=60$, an SVM classic), **lidar-return material classification**, or **IMU gait/anomaly detection** — any setting with rich features and scarce labels.

---

## 5. Pitfalls & Tips

- **Always scale your features.** SVMs measure distances; an unscaled feature with a huge numeric range (e.g. pressure in Pa vs a normalized ratio) silently dominates the kernel. Wrap `StandardScaler` in a `Pipeline` so scaling is fit on the training fold only.
- **Tune $C$ and $\gamma$ jointly with cross-validation.** They interact — a good $C$ at one $\gamma$ is wrong at another. Use `GridSearchCV` over a log grid (`C ∈ {0.1,1,10,100}`, `gamma ∈ {scale, 0.01, 0.1, 1}`). Default `gamma="scale"` is a sane starting point.
- **Don't reach for poly degree > 3.** High-degree polynomial kernels are numerically unstable and prone to wild extrapolation. If RBF can't do it, poly usually won't either.
- **SVMs scale poorly with sample count.** Training is roughly $O(n^2)$ to $O(n^3)$ in the number of samples, and `SVC` keeps support vectors in memory. Beyond ~50k–100k rows, switch to `LinearSVC` / `SGDClassifier(loss="hinge")`, or move to tree ensembles / nets.
- **Raw `SVC` scores are not probabilities.** `decision_function` returns signed margins, not calibrated probabilities. Set `probability=True` (adds Platt scaling, slower) or wrap with `CalibratedClassifierCV` if you need probabilities for thresholding.
- **SVM vs trees, the quick rule:** prefer **SVM** for small/medium data, many continuous features, and when you expect a smooth decision surface (signals, spectra, embeddings). Prefer **tree ensembles** (Lesson 06) for large tabular data, mixed categorical+numeric features, missing values, and when you want feature-importance interpretability with near-zero scaling fuss.

---

## 6. Check Your Understanding

**Q1.** Two boundaries both separate your training data with 100% accuracy. Why does the SVM prefer the one with the larger margin?

<details><summary>Answer</summary>
The larger margin is more robust to noise and generalizes better: a new point that lands a bit off from training data still falls on the correct side because of the buffer. Geometrically, maximizing the margin minimizes $\lVert\mathbf{w}\rVert$, which is L2 regularization — the wide-street boundary is the "simplest" one and least likely to overfit.
</details>

**Q2.** You delete all training points *except* the support vectors and retrain. How does the boundary change?

<details><summary>Answer</summary>
It does not change at all. The optimal $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ has $\alpha_i = 0$ for every non-support-vector, so only support vectors enter the solution. This is the source of the SVM's sparsity and robustness.
</details>

**Q3.** Your RBF SVM gets 100% train accuracy but 70% test accuracy. Which knob did you likely set too high, and which way do you turn it?

<details><summary>Answer</summary>
Overfitting — likely $\gamma$ too high (each support vector's influence is too local, producing a wiggly boundary that memorizes the training set), and/or $C$ too high (margin too tight). Decrease $\gamma$ and/or decrease $C$ to smooth/widen the boundary, then re-validate.
</details>

**Q4.** Why is the kernel called a "similarity function," and what does the RBF kernel return for two identical points vs two very distant points?

<details><summary>Answer</summary>
$K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z})$ is an inner product in the lifted space, and inner products measure alignment/similarity. For RBF, $K=\exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2)$: identical points give $K=1$ (max similarity), distant points give $K\to 0$. A prediction is a similarity-weighted vote over the support vectors.
</details>

**Q5.** You have 5 million labeled rows of mostly categorical tabular telemetry. Is an RBF `SVC` a good choice? What would you use instead?

<details><summary>Answer</summary>
No. Kernel `SVC` is $O(n^2)$–$O(n^3)$ in samples and chokes on millions of rows, and RBF doesn't handle categorical features gracefully. Use a gradient-boosted tree ensemble (Lesson 06) for big mixed-type tabular data, or `LinearSVC`/`SGDClassifier(loss="hinge")` if you specifically want a linear max-margin model at scale.
</details>

---

## Recap & Next

- The SVM finds the **maximum-margin** hyperplane — the widest empty street between classes — making it robust by construction.
- Only the **support vectors** (points on the margin) define the boundary; the soft-margin **hinge loss + L2** objective, governed by **$C$**, lets it tolerate overlap.
- The **kernel trick** swaps inner products for a **similarity function**, giving non-linear boundaries (linear / poly / RBF) for free; **$\gamma$** sets RBF boundary complexity.
- SVMs shine on **small, high-dimensional, continuous** data (signals, spectra, sonar); trees win on **large, mixed-type** tabular data.

Next we leave labeled data behind and let the model find structure on its own: **[08 — Unsupervised Learning: k-Means & PCA](08-kmeans-pca.md)**.
