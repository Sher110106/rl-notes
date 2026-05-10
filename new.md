# Actor-Critic Problem Set — Full Solutions

---

## Q1. One-step Actor-Critic with Linear Critic & Softmax Actor

### (a) Compute the TD Error δ

The TD error is defined as:

$$\delta = r + \gamma V(s', w) - V(s, w)$$

First, compute the state values:

$$V(s, w) = w^T \phi(s) = [0.4, 0.6, -0.1]^T [1, 0, 1]^T = (0.4)(1) + (0.6)(0) + (-0.1)(1) = 0.4 + 0 - 0.1 = 0.3$$

$$V(s', w) = w^T \phi(s') = [0.4, 0.6, -0.1]^T [0, 1, 1]^T = (0.4)(0) + (0.6)(1) + (-0.1)(1) = 0 + 0.6 - 0.1 = 0.5$$

Now plug in:

$$\delta = 4 + (0.9)(0.5) - 0.3 = 4 + 0.45 - 0.3 = \boxed{4.15}$$

---

### (b) Update the Critic Weights w

The critic update rule is:

$$w \leftarrow w + \alpha^w \cdot \delta \cdot \phi(s)$$

$$w \leftarrow [0.4, 0.6, -0.1]^T + (0.25)(4.15)[1, 0, 1]^T$$

$$= [0.4, 0.6, -0.1]^T + [1.0375, 0, 1.0375]^T$$

$$\boxed{w_{new} = [1.4375,\ 0.6,\ 0.9375]^T}$$

---

### (c) Compute ∇_θ log π(a1|s, θ) — Likelihood Ratio Trick

For a softmax policy over actions {a1, a2}, the log-probability is:

$$\log \pi(a|s, \theta) = \theta_a - \log\left(\sum_{a'} e^{\theta_{a'}}\right)$$

The gradient with respect to θ_i (the i-th logit) is:

$$\frac{\partial \log \pi(a|s,\theta)}{\partial \theta_i} = \mathbf{1}[a = a_i] - \pi(a_i|s,\theta)$$

This is the **likelihood ratio / score function** result. Written as a vector:

$$\nabla_\theta \log \pi(a|s,\theta) = \mathbf{e}_a - \boldsymbol{\pi}(s,\theta)$$

where **e_a** is the one-hot vector for the taken action.

For action **a1** with π(a1|s) = 0.65 and π(a2|s) = 0.35:

$$\nabla_\theta \log \pi(a1|s,\theta) = \begin{bmatrix}1\\0\end{bmatrix} - \begin{bmatrix}0.65\\0.35\end{bmatrix} = \boxed{\begin{bmatrix}0.35\\-0.35\end{bmatrix}}$$

**Intuition:** The gradient is positive for the taken action's logit (push it up) and negative for all others (pull them down), weighted by how surprising the choice was.

---

### (d) Update the Actor Parameters θ

The actor update rule is:

$$\theta \leftarrow \theta + \alpha^\theta \cdot \delta \cdot \nabla_\theta \log \pi(a1|s,\theta)$$

$$\theta \leftarrow \begin{bmatrix}0.8\\0.2\end{bmatrix} + (0.15)(4.15)\begin{bmatrix}0.35\\-0.35\end{bmatrix}$$

$$= \begin{bmatrix}0.8\\0.2\end{bmatrix} + (0.6225)\begin{bmatrix}0.35\\-0.35\end{bmatrix}$$

$$= \begin{bmatrix}0.8\\0.2\end{bmatrix} + \begin{bmatrix}0.2179\\-0.2179\end{bmatrix}$$

$$\boxed{\theta_{new} = \begin{bmatrix}1.0179\\-0.0179\end{bmatrix}}$$

---

### (e) New Policy Probabilities After Update

Using the softmax formula:

$$\pi(a1|s,\theta_{new}) = \frac{e^{\theta_1}}{e^{\theta_1} + e^{\theta_2}} = \frac{e^{1.0179}}{e^{1.0179} + e^{-0.0179}}$$

Compute:

- e^{1.0179} ≈ 2.7672
- e^{-0.0179} ≈ 0.9823
- Sum = 2.7672 + 0.9823 = 3.7495

$$\pi(a1|s) = \frac{2.7672}{3.7495} \approx \boxed{0.738}$$

$$\pi(a2|s) = \frac{0.9823}{3.7495} \approx \boxed{0.262}$$

The probability of a1 increased from 0.65 to 0.738, which makes sense because the TD error was large and positive (the outcome was better than expected), so the actor correctly increased the probability of the action that led to that good outcome.

---
---

## Q2. Gaussian Actor-Critic (Continuous Action)

### Step 1: Current Mean μ

$$\mu = \theta^T \phi(s) = [1.2, 0.5]^T [0.5, 0.8]^T = (1.2)(0.5) + (0.5)(0.8) = 0.6 + 0.4 = \boxed{1.0}$$

---

### Step 2: TD Error δ

First compute state values:

$$V(s,w) = w^T \phi(s) = [0.3, 0.8]^T [0.5, 0.8]^T = (0.3)(0.5) + (0.8)(0.8) = 0.15 + 0.64 = 0.79$$

$$V(s',w) = w^T \phi(s') = [0.3, 0.8]^T [0.6, 0.7]^T = (0.3)(0.6) + (0.8)(0.7) = 0.18 + 0.56 = 0.74$$

$$\delta = r + \gamma V(s',w) - V(s,w) = 3.5 + (0.92)(0.74) - 0.79$$

$$= 3.5 + 0.6808 - 0.79 = \boxed{3.3908}$$

---

### Step 3: Update Critic Weights w

$$w \leftarrow w + \alpha^w \cdot \delta \cdot \phi(s)$$

$$= \begin{bmatrix}0.3\\0.8\end{bmatrix} + (0.2)(3.3908)\begin{bmatrix}0.5\\0.8\end{bmatrix}$$

$$= \begin{bmatrix}0.3\\0.8\end{bmatrix} + (0.67816)\begin{bmatrix}0.5\\0.8\end{bmatrix}$$

$$= \begin{bmatrix}0.3\\0.8\end{bmatrix} + \begin{bmatrix}0.33908\\0.54253\end{bmatrix}$$

$$\boxed{w_{new} = \begin{bmatrix}0.63908\\1.34253\end{bmatrix}}$$

---

### Step 4: Compute ∂ ln π(a|s,θ) / ∂θ

For a Gaussian policy π(a|s,θ) = N(μ, σ²) with μ = θ^T φ(s) and fixed σ:

$$\ln \pi(a|s,\theta) = -\frac{(a - \mu)^2}{2\sigma^2} + \text{const}$$

Taking the gradient with respect to θ and applying the chain rule:

$$\frac{\partial \ln \pi(a|s,\theta)}{\partial \theta} = \frac{(a - \mu)}{\sigma^2} \cdot \phi(s)$$

This is the **score function for a Gaussian** — it points in the direction of the feature vector, scaled by how far the taken action was from the current mean, normalized by variance.

Plugging in values: a = 2.4, μ = 1.0, σ = 0.3, σ² = 0.09:

$$\frac{\partial \ln \pi}{\partial \theta} = \frac{(2.4 - 1.0)}{0.09} \cdot \begin{bmatrix}0.5\\0.8\end{bmatrix} = \frac{1.4}{0.09} \cdot \begin{bmatrix}0.5\\0.8\end{bmatrix} = 15.556 \cdot \begin{bmatrix}0.5\\0.8\end{bmatrix}$$

$$= \boxed{\begin{bmatrix}7.778\\12.444\end{bmatrix}}$$

---

### Step 5: Update Actor Parameters θ

$$\theta \leftarrow \theta + \alpha^\theta \cdot \delta \cdot \nabla_\theta \ln \pi(a|s,\theta)$$

$$= \begin{bmatrix}1.2\\0.5\end{bmatrix} + (0.12)(3.3908)\begin{bmatrix}7.778\\12.444\end{bmatrix}$$

$$= \begin{bmatrix}1.2\\0.5\end{bmatrix} + (0.40690)\begin{bmatrix}7.778\\12.444\end{bmatrix}$$

$$= \begin{bmatrix}1.2\\0.5\end{bmatrix} + \begin{bmatrix}3.1649\\5.0649\end{bmatrix}$$

$$\boxed{\theta_{new} = \begin{bmatrix}4.3649\\5.5649\end{bmatrix}}$$

---

### Step 6: New Mean μ' After Update

$$\mu' = \theta_{new}^T \phi(s) = (4.3649)(0.5) + (5.5649)(0.8) = 2.1825 + 4.4519 = \boxed{6.634}$$

The mean shifted dramatically from 1.0 to 6.634 because the taken action (2.4) was much higher than the previous mean (1.0), and the TD error was large and positive, so the actor is strongly incentivized to push the mean upward toward higher-reward actions. In practice, smaller learning rates or gradient clipping would be used to prevent such large jumps.

---
---

## Q3. Multi-step Reasoning Actor-Critic

### (a) Calculate δ

$$V(s_t, w) = w^T \phi(s_t) = [2.0, 1.5]^T [1, 0]^T = 2.0$$

$$V(s_{t+1}, w) = w^T \phi(s_{t+1}) = [2.0, 1.5]^T [0, 1]^T = 1.5$$

$$\delta = r + \gamma V(s_{t+1}, w) - V(s_t, w) = 2 + (0.95)(1.5) - 2.0 = 2 + 1.425 - 2.0 = \boxed{1.425}$$

---

### (b) Update Critic w

$$w \leftarrow w + \alpha^w \cdot \delta \cdot \phi(s_t)$$

$$= \begin{bmatrix}2.0\\1.5\end{bmatrix} + (0.3)(1.425)\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}2.0\\1.5\end{bmatrix} + \begin{bmatrix}0.4275\\0\end{bmatrix}$$

$$\boxed{w_{new} = \begin{bmatrix}2.4275\\1.5\end{bmatrix}}$$

---

### (c) Expression for ∇_θ log π(a1|s, θ) — Softmax Policy

For a softmax policy with logit vector θ = [θ_1, θ_2]^T, the score function is:

$$\nabla_\theta \log \pi(a1|s,\theta) = \mathbf{e}_{a1} - \boldsymbol{\pi}(s,\theta)$$

where e_{a1} = [1, 0]^T is the one-hot encoding of action a1, and π(s,θ) is the vector of action probabilities.

Explicitly for each component:

$$\frac{\partial \log \pi(a1|s,\theta)}{\partial \theta_i} = \mathbf{1}[i=1] - \pi(a_i|s,\theta)$$

Substituting the given probabilities π(a1|s) = 0.817 and π(a2|s) = 0.183:

$$\nabla_\theta \log \pi(a1|s,\theta) = \begin{bmatrix}1\\0\end{bmatrix} - \begin{bmatrix}0.817\\0.183\end{bmatrix} = \boxed{\begin{bmatrix}0.183\\-0.183\end{bmatrix}}$$

---

### (d) Update θ

$$\theta \leftarrow \theta + \alpha^\theta \cdot \delta \cdot \nabla_\theta \log \pi(a1|s,\theta)$$

$$= \begin{bmatrix}1.0\\-0.5\end{bmatrix} + (0.25)(1.425)\begin{bmatrix}0.183\\-0.183\end{bmatrix}$$

$$= \begin{bmatrix}1.0\\-0.5\end{bmatrix} + (0.35625)\begin{bmatrix}0.183\\-0.183\end{bmatrix}$$

$$= \begin{bmatrix}1.0\\-0.5\end{bmatrix} + \begin{bmatrix}0.06519\\-0.06519\end{bmatrix}$$

$$\boxed{\theta_{new} = \begin{bmatrix}1.06519\\-0.56519\end{bmatrix}}$$

---

### (e) New Probability of Choosing a1 Under the Updated Policy

Using softmax on the new logits θ₁ = 1.06519, θ₂ = −0.56519:

- e^{1.06519} ≈ 2.9012
- e^{-0.56519} ≈ 0.5683
- Sum = 2.9012 + 0.5683 = 3.4695

$$\pi_{new}(a1|s) = \frac{2.9012}{3.4695} \approx \boxed{0.836}$$

The probability of a1 increased from 0.817 to 0.836. The improvement is modest because the TD error (1.425) was not very large and the old probability was already high, but the direction is correct — positive δ reinforces the chosen action.

---
---

## Q4. Comparative Question — Why Multiply by δ Instead of Raw Return G_t

### Why Not Use Raw Return G_t?

The vanilla REINFORCE policy gradient uses:

$$\nabla_\theta J(\theta) \propto \mathbb{E}\left[G_t \cdot \nabla_\theta \log \pi(A_t|S_t,\theta)\right]$$

While this is unbiased, G_t is the sum of all future discounted rewards and carries extremely high variance. This is because G_t fluctuates enormously across episodes, even for the same state-action pair, due to stochasticity in future transitions and rewards. High variance slows learning drastically and requires many samples to get a reliable gradient signal.

---

### Why Use δ Instead?

In one-step Actor-Critic, we replace G_t with the TD error:

$$\delta_t = r_t + \gamma V(s_{t+1}, w) - V(s_t, w)$$

The update becomes:

$$\theta \leftarrow \theta + \alpha^\theta \cdot \delta_t \cdot \nabla_\theta \log \pi(A_t|S_t,\theta)$$

---

### Brief Derivation: δ as an Advantage Estimate

The true policy gradient theorem states:

$$\nabla_\theta J(\theta) \propto \mathbb{E}\left[Q^\pi(S_t, A_t) \cdot \nabla_\theta \log \pi(A_t|S_t,\theta)\right]$$

We can subtract any baseline b(S_t) that does not depend on the action without changing the expectation (because E[b(S) ∇ log π] = 0 by the score function identity). Choosing b(S_t) = V(S_t), we get:

$$\nabla_\theta J(\theta) \propto \mathbb{E}\left[\underbrace{(Q^\pi(S_t, A_t) - V^\pi(S_t))}_{\text{Advantage } A^\pi(S_t, A_t)} \cdot \nabla_\theta \log \pi(A_t|S_t,\theta)\right]$$

The **advantage function** A^π(s,a) = Q^π(s,a) − V^π(s,a) measures how much better action a is compared to the average action in state s.

Now, the one-step TD error is:

$$\delta_t = r_t + \gamma V(s_{t+1}, w) - V(s_t, w)$$

Taking expectations:

$$\mathbb{E}[\delta_t | s_t, a_t] = r_t + \gamma \mathbb{E}[V(s_{t+1})] - V(s_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

Therefore, **δ_t is an unbiased one-sample estimate of the advantage function**. It answers the question: "Was this particular action in this particular state better or worse than what was expected on average?"

---

### Summary of Benefits

Using δ instead of G_t achieves three things simultaneously:

**Variance reduction**: V(s,w) acts as a learned baseline that removes the "common tide" of all returns, so only the relative quality of the action matters. This is the single biggest practical benefit.

**Online learning**: δ is computed using just one transition (r, s'), so updates happen at every step rather than waiting until episode end as REINFORCE requires.

**Advantage interpretation**: A positive δ means "this action was better than average — increase its probability." A negative δ means "this action was worse than average — decrease its probability." This is a more semantically meaningful signal than raw cumulative return.

The trade-off is that δ introduces bias because V(s,w) is an approximation, not the true value function. This is the classic bias-variance trade-off in reinforcement learning: Actor-Critic accepts some bias in exchange for dramatically lower variance and the ability to learn online.
