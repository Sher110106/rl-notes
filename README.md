# Reinforcement Learning: Lectures 20–24 — Complete In-Depth Notes

---

## LECTURE 20: Function Approximation and DQN

---

### 1. Why Function Approximation at All?

Before we even talk about neural networks or DQN, you need to understand *why* we need function approximation in the first place.

In the tabular RL methods you studied earlier — Q-learning, SARSA, Monte Carlo — we maintained a table where every single state (or state-action pair) had its own entry. If you had 100 states and 4 actions, your Q-table had 400 entries. That is perfectly manageable.

But think about real problems. If your state is the position of a robot arm described by joint angles with 6 degrees of freedom, each continuous, your state space is literally infinite. You cannot build a table for infinite states. Similarly, even for seemingly "small" problems like Atari games, the raw pixel state of a screen might be 84x84 pixels with 3 color channels — the number of possible unique frames is astronomically large.

So we need a way to *generalize*. When the agent visits state s and learns something, it should be able to apply that knowledge to nearby, similar states. A table cannot do this — each entry is independent. A function can do this — if the function is smooth, then states that are similar in their input features will have similar output values.

That is the core idea of function approximation: instead of a table, we maintain a *parameterized function* that maps states (or state-action pairs) to values, and we update the *parameters* of that function based on experience. The parameters are shared across all states, so learning in one region generalizes to nearby regions.

---

### 2. Semi-Gradient Methods

Let us be precise about how we actually do gradient-based updates in TD learning with function approximation.

Suppose we are doing Q-learning with a linear function approximator. Our Q-function is:

    Q(s, a) = Φ^T(s, a) × w

Here, Φ(s, a) is a *feature vector* — a fixed vector of numbers that describes the state-action pair. For example, if we have 4 features, Φ(s, a) is a 4-dimensional vector. The weight vector w is also 4-dimensional. The Q-value is just the dot product of these two vectors.

Now, the TD error in Q-learning is:

    δ_t = r_{t+1} + γ · max_a Q(s_{t+1}, a) − Q(s_t, a_t)

This is the difference between what we got (reward plus discounted future value) and what we predicted. We want to minimize the squared TD error:

    Loss = [r_{t+1} + γ · max_a Q(s_{t+1}, a) − Q(s_t, a_t)]²

If we take the gradient of this loss with respect to w, we get:

    ∇_w [Loss] = −2 · δ_t · Φ(s_t, a_t)

The update rule then becomes:

    w_{t+1} = w_t + α · δ_t · Φ(s_t, a_t)

This looks clean, but there is a subtle and important issue. When we computed the gradient, we treated the TD *target* — the term r_{t+1} + γ · max_a Q(s_{t+1}, a) — as a *constant*. But it is not a constant! It depends on w through Q(s_{t+1}, a).

If we were doing true gradient descent, we would differentiate through the target as well. But we do not. We pretend the target is fixed and only differentiate through the predicted value Q(s_t, a_t).

This is why it is called a **Semi-Gradient** method: we are computing only part of the true gradient. It is an approximation to gradient descent. The consequence is that semi-gradient methods do not have the same nice convergence guarantees as true gradient descent. In the nonlinear case (neural networks), they can sometimes diverge. But in practice they work very well, and they are much simpler to implement than computing the full gradient.

The intuition for why this is done is practical: the TD target is our *estimate* of the true value, and we want to move the prediction toward that estimate. We treat the target as a "teacher" signal — a fixed number we are trying to match — and we update w to reduce the prediction error. This is philosophically similar to supervised learning.

---

### 3. Feature Construction and State Aggregation

If we are using a linear function approximator, the quality of our approximation depends entirely on the quality of our features Φ(s, a). The features are doing all the heavy lifting — they are determining how states are represented. Let us study several approaches.

#### 3.1 Naïve State Space Aggregation (Grid-Based Discretization)

The simplest approach is to divide the continuous state space into a grid of cells and treat each cell as a single discrete state. Every point in the same cell gets the same feature representation.

For example, if your state is a 2D position (x, y), you might divide the space into a 10×10 grid. Any position in cell (3, 7) gets represented as "I am in cell (3, 7)." This is a one-hot encoding: a 100-dimensional vector where only the entry for cell (3, 7) is 1 and all others are 0.

This works but has a fatal flaw: two positions that are very close together but happen to fall in different cells get completely different representations, while two positions that might be far apart but in the same cell get identical representations. The boundaries of your grid cells are artificial — they do not reflect the true structure of the problem.

Also, if your grid is too coarse, you lose resolution. If it is too fine, you have the original tabular problem with too many states.

#### 3.2 Coarse Coding

Coarse coding is a smarter approach. Instead of hard grid boundaries, you use *receptive fields* — overlapping regions in state space. Each feature i corresponds to one receptive field (typically a circle in 2D). Feature i is 1 if the current state falls within receptive field i, and 0 otherwise.

The key insight is that because the receptive fields *overlap*, a given state activates multiple features simultaneously. Two nearby states will share many of the same active features, so their representations are similar, and their learned values will be similar. This gives you natural generalization.

Now there is an important trade-off:

**Large receptive fields**: When a receptive field is large, many states share it. This means learning transfers broadly across the state space — broad generalization. But you cannot make fine distinctions between states that are far apart from each other but share the same large receptive field.

**Small receptive fields**: When receptive fields are small, only very nearby states share them. This gives you fine discrimination — you can tell apart states that are close together. But generalization is narrow — you need a lot of data to learn about each part of the state space separately.

Crucially, there is an important nuance. The slides state: *"Initial generalization is controlled by the size and shape of the receptive fields, but acuity — the finest discrimination ultimately possible — is controlled more by the total number of features."*

What this means: after enough training (many data points), even large receptive fields can achieve fine resolution. This happens because many overlapping large receptive fields together can uniquely identify specific regions. The initial learning speed depends on field size, but the eventual precision depends on how many features you have in total.

This is demonstrated in Figure 9.8 in the slides: with 10 examples, all feature widths give rough approximations. With 10240 examples, even broad features converge to an accurate representation of the target function.

#### 3.3 Tile Coding

Tile coding is the most practical form of coarse coding for multi-dimensional continuous spaces. It is computationally efficient and gives you precise control over the resolution.

Here is how it works. You create multiple overlapping grids, called *tilings*. Each tiling covers the entire state space with a grid of tiles. Each tiling is offset from the others by a small amount.

For a given state, you identify which tile it falls in for *each* tiling. The feature vector is then the concatenation of these one-hot encodings — one entry per tile across all tilings.

In the example from the slides: if you have 4 tilings and each tiling has a 4×4 grid, then the total number of features is 4 × 4 × 4 = 64. For any given state, exactly 4 of these features are active (one from each tiling). All other 64 − 4 = 60 features are 0.

One important practical advantage of tile coding: **the number of active features is always the same** (equal to the number of tilings), regardless of which state you are in. This means the step size α has a consistent effect across all states, which makes learning more stable. In contrast, with irregular features, some states might activate many features and others very few, creating inconsistent learning rates.

The comparison in Figure 9.10 shows tile coding with 50 tilings converges to much lower error than state aggregation with a single tiling, on the same number of episodes.

#### 3.4 Multi-Resolution Feature Aggregation

This is research work from Professor Manjanna. The idea is to use features at multiple *scales simultaneously*. Instead of having all features at the same resolution, you have some features representing large regions (coarse) and some representing small regions (fine).

This is analogous to how humans perceive space: you know roughly which neighborhood you are in (coarse), and also exactly which room of a house you are in (fine). Both levels of information are useful simultaneously.

This approach achieves higher rewards than uniform-grid aggregation and is computationally more efficient than all-grid aggregation, especially as the world size grows.

#### 3.5 The State-Action Feature Vector Φ(s, a)

When we need features for state-action pairs rather than just states, a common trick is:

    Φ(s, a) = Φ'(s) ⊗ δ_{a, a'}

Here, Φ'(s) is the state feature vector, and δ_{a, a'} is the Kronecker delta (1 if a equals action a', 0 otherwise). The tensor product ⊗ means: if there are k state features and m actions, we create a vector of k × m entries. Only the k entries corresponding to the actual action a are filled with the state features; all other entries are 0.

This is shown in the slides with an 80-state grid and 4 actions. The state vector Φ'(s) has 80 entries (one per cell). The action vector δ_{aa'} has 4 entries (one per action). The combined feature vector Φ(s,a) has 320 entries, with only 80 nonzero (those corresponding to the current action).

---

### 4. Stochastic Gradient Descent in RL

In standard (batch) gradient descent, you compute the gradient using all data points and then take one step. In Stochastic Gradient Descent (SGD), you approximate the true gradient using a single randomly selected data point.

In RL, our "data points" are transitions (s, a, r, s'). We process each transition as it arrives and update our parameters based on that single transition. This is why RL naturally uses SGD: we update after each step, not after collecting all data.

In practice, **Mini-Batch Gradient Descent** is used: you collect a small batch of transitions (say 32 or 64) and compute the gradient averaged over that batch. This gives a better estimate of the gradient than a single transition, while being more computationally efficient than processing the entire dataset. This is still called SGD in the broader sense.

The advantage of mini-batch over single-sample SGD: lower variance in the gradient estimate, smoother convergence. The advantage over batch: you update frequently and can start learning with limited data.

---

### 5. Non-linear Function Approximation: Neural Networks

Linear function approximators have a fundamental limitation: they can only represent *linear functions* of the features. No matter how clever your features are, the output Q(s, a) = Φ^T(s, a) × w is a weighted sum of features. Some value functions are inherently nonlinear and cannot be well-approximated this way.

Neural networks are nonlinear function approximators. They are parameterized functions of the form:

    f(x; θ) = output

where x is the input (state or state-action pair) and θ represents all the weights and biases in the network. The output is computed through multiple layers of linear transformations followed by nonlinear activation functions (like ReLU — Rectified Linear Unit, which outputs max(0, x)).

Key properties:

1. **Nonlinearity**: The composition of linear layers and nonlinear activations allows neural networks to represent arbitrarily complex functions. Theoretically, a sufficiently wide two-layer neural network can approximate any continuous function.

2. **Feature learning**: Unlike tile coding or coarse coding, neural networks *learn* their features automatically from data. The early layers of the network learn to extract relevant patterns from raw input; later layers combine these into value estimates. You do not need to handcraft features.

3. **Generalization**: Because the weights are shared across inputs (through the same network), learning about one state naturally influences predictions for similar states.

4. **Fully differentiable**: Every operation in a neural network is differentiable. This means we can compute the gradient of the output with respect to all parameters θ using backpropagation, and use gradient descent to update θ.

5. **Disadvantage: Data and compute hungry**. Neural networks require large amounts of data and significant computation to train well. They also have many hyperparameters (architecture, learning rate, etc.) that need tuning.

The historical example of **TD-Gammon** (Tesauro, 1992) showed that even before the deep learning era, combining TD learning with a neural network (with 40–80 hidden units) allowed an agent to learn to play backgammon at master level purely by playing against itself, starting from random weights. The input was 198 features describing the board state, and the output was the predicted probability of winning.

---

### 6. Control with Function Approximation

When doing control (not just prediction), we use a similar alternating procedure to policy iteration:

**Policy Evaluation**: Approximate the action-value function Q_π(s, a) ≈ q̂(s, a; w) using function approximation with the current policy.

**Policy Improvement**: Use ε-greedy policy improvement based on the current approximation q̂.

This alternation is depicted in the classic diagram from David Silver's course: starting from a point, we alternately move toward the "true value of current policy" line (policy evaluation) and the "greedy policy improvement" line, converging toward the optimal policy Q*.

---

### 7. Deep Q-Learning Network (DQN)

DQN, introduced by DeepMind in the Nature 2015 paper, was the breakthrough that demonstrated a single agent could learn to play 49 Atari games directly from raw pixels, achieving superhuman performance on many.

The architecture:

The input is a stack of 4 consecutive game frames, each preprocessed to 84×84 grayscale pixels. So the raw input is 84 × 84 × 4.

The network has:
- A convolutional layer with 16 8×8 filters (for extracting spatial features from pixels)
- A second convolutional layer with 32 4×4 filters
- A fully connected layer with 256 hidden units
- An output fully connected layer with one output per action (18 actions for the Atari games)

The convolutional layers act as the **feature learning** part — they are extracting meaningful visual features from raw pixels automatically. The fully connected layers act as linear function approximators on top of those learned features.

The output is: for a given input frame stack (representing the current state), the network outputs Q-values for all 18 possible actions simultaneously. This is more efficient than evaluating each action separately.

**What is ReLU?** ReLU (Rectified Linear Unit) is the activation function used throughout the network. It is defined as max(0, x). It is used because it is computationally cheap, avoids the vanishing gradient problem, and works very well empirically.

---

## LECTURE 21: DQN and Policy Gradient

---

### 8. Q-Network Learning and Its Problems

The basic Q-network update rule is:

    w_{t+1} = w_t − (α/2) · ∇_w [r_{t+1} + γ · max_a q̂(s_{t+1}, a) − q̂(s_t, a_t)]²

This looks like standard semi-gradient TD. But when you use a neural network, two major problems cause the learning to **diverge** (the Q-values explode to infinity or oscillate wildly):

**Problem 1: Correlations between samples**

When the agent moves through the environment step by step, consecutive transitions (s_t, a_t, r_{t+1}, s_{t+1}) and (s_{t+1}, a_{t+1}, r_{t+2}, s_{t+2}) are highly correlated. The states are nearly identical (one step apart). 

Standard SGD assumes that training samples are independently drawn. If you violate this and train on highly correlated sequential data, the network gets "stuck" in local patterns and fails to generalize properly. The gradient updates from correlated samples all push in the same direction temporarily, causing the network weights to oscillate.

**Problem 2: Non-stationarity of targets**

In supervised learning, your targets (labels) are fixed. In Q-learning with function approximation, the target is:

    r_{t+1} + γ · max_a q̂(s_{t+1}, a; w)

This target *depends on w*. So every time you update w, all your targets change too. You are chasing a moving target. This is like trying to hit a bullseye while someone is moving the dartboard at the same time. This non-stationarity makes convergence very difficult.

**Solutions: Replay Memory and Freeze Target Network**

These are the two key innovations of DQN. They are heuristics — they work empirically but do not come with theoretical guarantees.

---

### 9. Experience Replay (Replay Memory)

The idea is simple and elegant. Instead of training on transitions as they arrive, you store every transition (s_t, a_t, r_{t+1}, s_{t+1}) in a large buffer called the **replay memory**. After each step, you sample a *random mini-batch* from this buffer and train on that.

Why does this help?

1. **Breaking correlations**: By randomly sampling from the buffer, the training mini-batch contains transitions from many different time points and many different parts of the state space. These are much less correlated than consecutive transitions. This is why the slides say: "A large replay buffer will result in less correlation."

2. **Data efficiency**: Each transition can be used for training multiple times (since it stays in the buffer). This is much more data-efficient than using each transition once and discarding it.

The buffer has a maximum capacity (say, 1 million transitions). When full, old transitions are overwritten by new ones. This also helps somewhat with non-stationarity, as the buffer always contains recent experience.

---

### 10. Freeze Target Network

To address the non-stationarity of targets, DQN maintains *two* separate networks with the same architecture:

1. **Online network** with weights w: This is the network that is updated during training. It is used to select actions.

2. **Target network** with weights w⁻: This is a *frozen copy* of the online network. It is used to compute the TD target.

The loss becomes:

    [r_{t+1} + γ · max_a q̂(s_{t+1}, a; w⁻) − q̂(s_t, a_t; w)]²

The target network weights w⁻ are held *fixed* for a large number of steps (e.g., 10,000 steps). Then they are updated to the current online network weights w, and frozen again.

Why does this help? Because the targets are now computed by the frozen target network, they do not change every step. The target network is like a fixed "teacher" — it changes slowly and infrequently, giving the online network a stable target to train toward.

Together, experience replay and the frozen target network make DQN stable and convergent in practice.

---

### 11. DQN Summary: Architecture Details

- Input: 4 stacked 84×84 grayscale frames
- Conv layer 1: 16 filters of size 8×8, ReLU
- Conv layer 2: 32 filters of size 4×4, ReLU
- Fully connected: 256 units, ReLU
- Output: |A| units (one Q-value per action), linear (no activation)
- Loss: **Huber loss** (not mean squared error)

**Huber Loss vs. Least Squares Loss**:

The regular MSE loss squares the error: L = (1/2)a². For large errors, this grows quadratically — a single very large TD error can dominate the gradient and cause a massive weight update, destabilizing training.

The Huber loss is:
- L(a) = (1/2)a²    if |a| ≤ δ
- L(a) = δ(|a| − δ/2)    otherwise

For small errors, it behaves like squared loss (quadratic). For large errors, it switches to linear loss, which gives a constant gradient. This prevents any single large error from causing an explosion in the weight update. It makes training more robust to outliers.

---

### 12. Double Q-Learning

There is a well-known problem with standard Q-learning: it **overestimates** action values. Why?

The max operator in the TD target selects the action with the highest estimated Q-value:

    max_a Q(s', a)

But our Q-value estimates are noisy — they have random errors. The maximum of several noisy estimates is systematically biased upward (this is a statistical fact — the max of noisy values is biased high). Over time, these overestimates compound and the Q-values become inflated, causing the agent to be overconfident about certain actions.

The solution is **Double Q-Learning**: use one Q-function to *select* the best action, and a *different* Q-function to *evaluate* that action.

In tabular Double Q-Learning, we maintain two separate Q-functions Q1 and Q2. With 50% probability:
- Use Q1 to select the action: a* = argmax_a Q1(s', a)
- Use Q2 to evaluate it: target = R + γ · Q2(s', a*)

This decoupling of selection and evaluation removes the systematic bias, because the estimation errors in Q1 and Q2 are (approximately) independent.

In **Double DQN**, we use the online network to select the action and the target network to evaluate it:

    Y_t^DoubleDQN = R_{t+1} + γ · Q(S_{t+1}, argmax_a Q(S_{t+1}, a; θ_t), θ_t⁻)

The online network θ_t selects which action is best. The target network θ_t⁻ evaluates the value of that action. Since the two networks are not synchronized (target network is frozen), their errors are less correlated, reducing the overestimation bias.

---

### 13. DQN Performance on Atari

The DQN results show impressive performance across 49 games, often achieving scores far above human level (e.g., 2539% of human performance on Video Pinball). However, it performs poorly on some games (below human level on, for example, Private Eye, Gravitar). The games where DQN struggles tend to require long-term planning or have sparse rewards.

---

### 14. Introduction to Policy Search Methods

So far, everything we have done is *value-based*: we learn a value function (V or Q) and derive a policy from it (e.g., ε-greedy). Policy search methods take a completely different approach: they directly search in the *space of policies*, without necessarily maintaining a value function.

**Why policy search?**

1. **Simpler description**: For some problems, directly parameterizing the policy is more natural than learning a value function.

2. **Better convergence**: Policy gradient methods are guaranteed to converge to at least a local optimum under standard conditions, whereas value-based methods with function approximation can diverge.

3. **Robust to partial observability**: Since policy search does not rely on a state-based value function, it does not require a Markovian state. This makes it more applicable to partially observable environments.

4. **Handles continuous action spaces naturally**: If your action space is continuous (e.g., joint torques for a robot arm), you cannot easily do argmax over actions. Policy gradient methods can directly output continuous actions.

5. **Smooth action probability changes**: With a continuous policy parameterization, a small change in θ leads to a small change in action probabilities. This is more stable than ε-greedy, where a small change in Q-values can completely change the greedy action.

Two main approaches to policy search:
- **Direct policy search / Evolutionary methods**: Use genetic algorithms or other evolutionary strategies to search the parameter space.
- **Policy Gradient methods**: Use gradient ascent on the expected return to update policy parameters.

---

### 15. Policy Gradient Methods: The Setup

We parameterize the policy as π(a|s, θ): a probability distribution over actions, conditioned on the state, parameterized by θ.

A common parameterization is the **softmax policy** (for discrete actions):

    π(a|s, θ) = exp(h(s, a, θ)) / Σ_b exp(h(s, b, θ))

where h(s, a, θ) = θ^T · x(s, a) is the "preference" for action a in state s. This is essentially a softmax over learned preferences. As θ changes, action probabilities change smoothly.

For continuous action spaces, a common choice is a **Gaussian policy**:

    π(a|s, θ) = (1/√(2πσ²)) · exp(−(a − θ^T Φ_s)² / (2σ²))

The mean of the Gaussian is θ^T Φ_s (a linear function of state features), and σ is the standard deviation. The agent samples actions from this Gaussian.

The **objective function** J(θ) measures the performance of policy π_θ:

    J(θ) = E[r_t] = Σ_a q*(a) π_θ(a)    [for bandits]
    J(θ) = v_{π_θ}(s_0)    [for full MDPs, starting from s_0]

We want to *maximize* J(θ) with respect to θ. We do this using **gradient ascent**:

    θ ← θ + α · ∇J(θ)

The challenge is computing ∇J(θ). This is what the policy gradient theorem addresses.

---

### 16. Basic Policy Gradient: Computing the Gradient

For simplicity, consider the bandit setting (single state). We have:

    J(θ) = Σ_a q*(a) π_θ(a)

Taking the gradient:

    ∇J(θ) = Σ_a q*(a) · ∇π_θ(a)
           = Σ_a q*(a) · [∇π_θ(a) / π_θ(a)] · π_θ(a)
           = Σ_a π_θ(a) · q*(a) · [∇π_θ(a) / π_θ(a)]
           = E_{π_θ}[q*(a) · ∇π_θ(a) / π_θ(a)]

This is the **likelihood ratio trick** (also called the log-derivative trick):

    ∇π_θ(a) / π_θ(a) = ∇ log π_θ(a)

So:

    ∇J(θ) = E_{π_θ}[q*(a) · ∇ log π_θ(a)]

We can estimate this expectation from N samples:

    ∇̂J(θ) = (1/N) Σ_{i=1}^{N} r_i · ∇π_θ(a_i) / π_θ(a_i)

This is a Monte Carlo estimate of the gradient. We collect N rollouts, and for each one, we weight the gradient of log policy by the reward.

The intuition is beautiful: we are pushing the policy parameters in the direction that increases the log-probability of actions that led to high rewards, and decreasing the log-probability of actions that led to low rewards. This is "reward-weighted regression."

---

## LECTURE 22: Policy Gradient Theorem

---

### 17. REINFORCE Algorithm

REINFORCE (Williams, 1992) is the foundational Monte Carlo policy gradient algorithm for the full MDP setting (not just bandits).

The algorithm:

1. Initialize θ arbitrarily.
2. Loop forever:
   a. Generate a full episode S_0, A_0, R_1, S_1, ..., S_{T-1}, A_{T-1}, R_T following π_θ.
   b. For each time step t = 0, 1, ..., T-1:
      - Compute the return: G_t = Σ_{k=t+1}^{T} γ^{k-t-1} R_k
      - Update: θ ← θ + α · γ^t · G_t · ∇ log π_θ(A_t | S_t)

The update at each time step is:

    Δθ_t = α · r_t · ∇π_θ(a_t) / π_θ(a_t) = α · r_t · ∂ log π_θ(a_t) / ∂θ

Properties of REINFORCE:
- Computes an **unbiased estimate** of the gradient ∇J(θ). This is because we are using actual sampled returns, not bootstrapped estimates. The expected value of the Monte Carlo gradient estimate equals the true gradient.
- **Very high variance**. The return G_t is one sample of a random variable that can vary enormously. Some episodes go well by luck, others by misfortune. This variance makes learning slow and unstable.
- The variance is related to the episode length. Longer episodes and larger state spaces lead to higher variance.

---

### 18. REINFORCE with Baseline

We can dramatically reduce variance (without introducing bias) by subtracting a *baseline* b(S_t) from the return:

    Δθ_t = α · (G_t − b(S_t)) · ∂ log π_θ(A_t | S_t) / ∂θ

The baseline b(S_t) can be any function of the state — it just cannot depend on the action A_t. Subtracting a state-dependent baseline does not change the expected value of the gradient update (it is an unbiased modification), but it can greatly reduce variance.

**Why does it not change the expected value?**

Because:

    E_{A_t ~ π_θ}[b(S_t) · ∇ log π_θ(A_t | S_t)] = b(S_t) · Σ_a ∇π_θ(a|S_t) = b(S_t) · ∇[Σ_a π_θ(a|S_t)] = b(S_t) · ∇1 = 0

The sum of all action probabilities is always 1, so its gradient is always 0. Therefore, subtracting any state-dependent baseline from G_t does not affect the expected gradient.

**What baseline to use?** The most common choice is the estimated state value function: b(S_t) = V̂(S_t). Then G_t − b(S_t) ≈ G_t − V_π(S_t) is an estimate of the **advantage** of action A_t — how much better this action was compared to the average action in state S_t. Positive advantage = we got more than expected, increase probability. Negative = we got less, decrease probability.

Two important constraints:
- The baseline b_t must not depend on the action a_t (or it would bias the gradient).
- The learning rate α_t must not depend on r_t (or the scaling of the update would vary with reward magnitude).

The update rule with baseline:

    θ_{t+1} ≐ θ_t + α(G_t − b(S_t)) · ∂ log π_θ(A_t | S_t) / ∂θ

---

### 19. The Policy Gradient Theorem (Full MDP)

For the full episodic MDP, the policy gradient theorem states:

    ∇J(θ) ∝ Σ_s μ(s) Σ_a q_π(s, a) · ∇π(a|s, θ)

where μ(s) is the on-policy state distribution under π_θ.

The theorem is remarkable because computing ∇J(θ) naively requires differentiating through the entire transition dynamics of the MDP — including how the state distribution changes when θ changes. The policy gradient theorem says we *do not need to worry about that*. The gradient depends on the state distribution μ but not its derivatives.

Let us trace through the proof as shown in the slides (Panel-by-Panel):

**Panel 1 (Setup)**:
- Given parameterized policy π_θ
- Performance w.r.t. θ is J(θ)
- Every episode starts at state S_0
- J(θ) can be defined as V_{π_θ}(S_0) = the value function of the starting state
- J(θ) = ∫_τ P_θ(τ) · r(τ) dτ
  where τ is a trajectory, P_θ(τ) is the probability of that trajectory under π_θ, and r(τ) is the cumulative reward.

**Panel 2 (Gradient)**:
- The gradient ascent for parameter θ is: θ_{k+1} = θ_k + α · ∇_θ J(θ)
- ∇_θ J(θ) = ∫ ∇_θ P_θ(τ) · r(τ) dτ
- Using the identity ∇P_θ(τ) = P_θ(τ) · ∇ log P_θ(τ):
  ∇_θ J(θ) = ∫ P_θ(τ) · ∇_θ log P_θ(τ) · r(τ) dτ
             = E_τ[∇_θ log P_θ(τ) · r(τ)]

**Panel 3 (Decomposing log P_θ(τ))**:
- A trajectory τ has probability: P_θ(τ) = P(s_0) · Π_{t=0}^{H-1} π_θ(a_t|s_t) · P(s_{t+1}|s_t, a_t)
- Taking log of both sides:
  log P_θ(τ) = log P(s_0) + Σ_{t=0}^{H-1} log π_θ(a_t|s_t) + Σ_{t=0}^{H-1} log P(s_{t+1}|s_t, a_t)
- Differentiating w.r.t. θ: the log P(s_0) term is 0 (doesn't depend on θ). The log P(s_{t+1}|s_t, a_t) terms are 0 (transition probabilities don't depend on θ). Only the policy terms survive:
  ∇_θ log P_θ(τ) = Σ_{t=0}^{H-1} ∇_θ log π_θ(a_t|s_t)

**Panel 4 (Final Expression)**:
- Substituting back:
  ∇_θ J(θ) = E_τ[Σ_{t=0}^{H-1} ∇_θ log π_θ(a_t|s_t) · Σ_{j=0}^{H-1} r(s_j, a_j)]
- The key insight is that future rewards (after time t) do not depend on past actions (before time t), so we can use only future rewards:
  ∇_θ J(θ) = (1/m) Σ_{i=1}^{m} [Σ_{t=0}^{H-1} ∇_θ log π_θ(a_t^(i)|s_t^(i)) · Σ_{j=t}^{H-1} r(s_j^(i), a_j^(i)) − b(s_t^(i))]

where m is the number of trajectories sampled and the superscript (i) denotes the i-th trajectory.

This is the **practical implementation formula** for policy gradient methods. For each trajectory, for each time step, we compute the gradient of the log-policy and weight it by the future returns from that time step onward (minus a baseline).

---

### 20. The Likelihood Ratio Trick Revisited

The key mathematical identity used throughout policy gradient derivations is:

    ∇p_θ(y) = p_θ(y) · ∇ log p_θ(y)

This follows from the chain rule: ∇ log f = ∇f / f, so ∇f = f · ∇ log f.

Why is this trick useful? Because the expectation of ∇ log π_θ(a|s) under π_θ can be estimated by sampling: we just sample trajectories from π_θ and compute ∇ log π_θ for each observed action. We do not need to differentiate through the environment dynamics.

The final practical gradient estimator is:

    ∇_θ J_θ = (1/m) Σ_{i=1}^{m} Σ_{t=0}^{H-1} ∇_θ log π_θ(a_t^(i)|s_t^(i)) · [Σ_{j=t}^{H-1} r(s_j^(i), a_j^(i)) − b(s_t^(i))]

---

## LECTURE 23: Actor-Critic Methods

---

### 21. Motivation for Actor-Critic

REINFORCE has a fundamental weakness: **high variance**. Because we use the full Monte Carlo return G_t as our estimate of q_π(S_t, A_t), and G_t is a single noisy sample, the gradient estimates are very noisy. This makes learning slow.

Recall that:

    E_π[G_t | S_t, A_t] = q_π(S_t, A_t)

So G_t is an unbiased estimate of q_π(S_t, A_t). The problem is its variance.

What if instead we used a *learned estimate* of q_π(S_t, A_t)? Such an estimate would have lower variance (because it averages over many possible futures implicitly), at the cost of introducing some bias (because our estimate is not perfect).

This is exactly what actor-critic methods do:

**Actor-Critic = Policy Gradient + Function Approximation for the Critic**

- **Actor**: The policy π_θ, updated by gradient ascent.
- **Critic**: An estimate v̂(s, w) of the state-value function, updated by TD learning.

The actor takes actions. The critic evaluates those actions by estimating their value. The critic's evaluation is used to update the actor.

---

### 22. Actor-Critic Update Rules

The update rule for the actor (policy):

    θ_{t+1} = θ_t + α(G_{t:t+1} − v̂(S_t, w)) · ∇π(A_t|S_t, θ_t) / π(A_t|S_t, θ_t)

where G_{t:t+1} = R_{t+1} + γ · v̂(S_{t+1}, w) is the one-step TD return.

Expanding:

    θ_{t+1} = θ_t + α(R_{t+1} + γ · v̂(S_{t+1}, w) − v̂(S_t, w)) · ∇π(A_t|S_t, θ_t) / π(A_t|S_t, θ_t)

The term R_{t+1} + γ · v̂(S_{t+1}, w) − v̂(S_t, w) is the **TD error δ_t**, which serves as the critic's evaluation of the actor's action.

    θ_{t+1} = θ_t + α · δ_t · ∇π(A_t|S_t, θ_t) / π(A_t|S_t, θ_t)
             = θ_t + α · δ_t · ∇ log π(A_t|S_t, θ_t)

The critic is updated using standard TD(0):

    w_{t+1} = w_t + α^w · δ_t · ∇ v̂(S_t, w)

---

### 23. The One-Step Actor-Critic Algorithm

The complete algorithm:

1. Initialize policy parameter θ and value weights w (e.g., to 0).
2. Loop forever (for each episode):
   - Initialize state S (first state of episode)
   - Loop while S is not terminal:
     a. Sample action A ~ π(·|S, θ)
     b. Take action A, observe reward R and next state S'
     c. Compute TD error: δ = R + γ · v̂(S', w) − v̂(S, w)
        [If S' is terminal, set v̂(S', w) = 0]
     d. Update critic: w ← w + α^w · δ · ∇ v̂(S, w)
     e. Update actor: θ ← θ + α^θ · δ · ∇ log π(A|S, θ)
     f. S ← S'

Note: two separate learning rates, one for the actor (α^θ) and one for the critic (α^w). These may need to be tuned separately.

---

### 24. Comparison with REINFORCE

Why is actor-critic better than REINFORCE?

In REINFORCE with baseline:

    θ_{t+1} ≐ θ_t + α(G_t − b(S_t)) · ∂ log π_θ(A_t|S_t) / ∂θ

The term G_t is the actual return from time t onward. It is **unbiased** (expected value equals q_π(S_t, A_t)) but has **high variance** because it depends on the randomness of all future steps.

In actor-critic:
- We replace G_t with R_{t+1} + γ · v̂(S_{t+1}, w) (one-step bootstrap)
- We use v̂(S_t, w) as the baseline

This introduces **bias** (because v̂ is not the true value function) but **reduces variance** dramatically. The TD error δ_t = R_{t+1} + γ · v̂(S_{t+1}, w) − v̂(S_t, w) depends only on one step of randomness (the immediate reward R_{t+1}), not the entire future trajectory.

The bias-variance trade-off:
- REINFORCE: unbiased, high variance → slow, unstable learning
- Actor-Critic (one-step): biased, low variance → faster, more stable learning

In the one-step AC algorithm, we use v̂ for *both* estimating q_π(S_t, A_t) (through bootstrapping) and as the baseline. This double use is what makes it efficient.

---

### 25. The Advantage Function

Define the **advantage function**:

    A_π(s, a) = q_π(s, a) − v_π(s)

This measures: how much better is action a in state s compared to the average action (according to policy π)?

- If A_π(s, a) > 0: action a is better than average → increase its probability
- If A_π(s, a) < 0: action a is worse than average → decrease its probability
- If A_π(s, a) = 0: action a is exactly average

The TD error serves as an *estimate* of the advantage function:

    δ_π(s, a) = q_π(s, a) − v_π(s) ≈ A_π(s, a)

This is not an exact equality, but in expectation: E[δ_t | S_t, A_t] = q_π(S_t, A_t) − v_π(S_t) = A_π(S_t, A_t).

The advantage function is a central concept in modern policy gradient methods. Using advantage estimates (rather than raw returns or Q-values) leads to more stable training because it is centered around zero.

---

### 26. The Full Actor-Critic Procedure

Step by step:

1. Take action a ~ π_θ(a|s) and receive (s, a, s', r)
2. Update the value parameter w using data (s, y) where y = r + γ · v̂(s', w):
   - Minimize the squared loss L(w) = (1/N) Σ_i ||v̂(s_i, w) − y_i||²
   - This step usually happens in batches: collect multiple data points and minimize the squared loss
3. Compute the TD error: δ̂(s, a) = r + γ · v̂(s', w) − v̂(s, w)
4. Update the policy: θ ← θ + α · δ̂(s, a) · ∇_θ log π_θ(a|s)
5. Return to step 1

The critic update (step 2) in batch mode: we collect N transitions, form targets y_i = r_i + γ · v̂(s_i', w), and minimize the squared error. This is standard supervised regression.

---

### 27. Asynchronous Advantage Actor-Critic (A3C)

A3C (Mnih et al., 2016) is a major practical advance in actor-critic methods. The key insight: instead of one agent interacting with one environment, run *multiple agents in parallel*, each with their own copy of the environment.

**Architecture**:
- A global set of shared parameters (θ, w) for the policy and value function.
- Multiple actor-learner threads (agents), each with a thread-specific copy (θ', w') of the parameters.

**Per-thread algorithm**:
1. Reset gradients dθ = 0, dw = 0.
2. Synchronize thread-specific parameters with global: θ' = θ, w' = w.
3. Gather experience by acting in the environment for up to t_max steps (or until terminal).
4. Compute the return R for the last state (0 if terminal, v̂(S_t, w') otherwise — bootstrap).
5. For each step backward from the last to the first:
   - R ← r_i + γ · R
   - Accumulate policy gradient: dθ ← dθ + ∇_{θ'} log π(a_i|s_i; θ') · (R − V(s_i; w'))
   - Accumulate value gradient: dw ← dw + ∂(R − V(s_i; w'))² / ∂w'
6. Update global parameters *asynchronously*: θ ← θ + dθ, w ← w + dw.

**Why does this work?**

Each agent explores different parts of the state space and generates independent experience. When you aggregate updates from multiple agents, you break the correlations between samples — similar to experience replay in DQN, but without needing to store a replay buffer.

The "A" in A3C stands for Asynchronous: different agents update the global parameters independently, without waiting for each other. This allows GPU-free parallelism (running on many CPU cores).

**Advantage**: A3C is fast (parallel), does not require a replay memory, and naturally explores diverse parts of the state space.

**Disadvantage**: Asynchronous updates can be stale — by the time a thread's gradients are applied to the global parameters, the global parameters may have changed significantly. This "staleness" introduces noise but is generally tolerable.

---

### 28. Synchronous Advantage Actor-Critic (A2C)

A2C removes the asynchronous part of A3C. All agents (running in parallel) gather experience simultaneously, and then a *coordinator* waits for all of them to finish before computing the gradient update.

**Architecture**:
- Global parameters (θ, w).
- Multiple agents gathering experience in parallel.
- A coordinator that collects all experience and performs a synchronized global update.

**Key difference from A3C**: Updates are synchronized — all agents use the same version of the parameters, and all gradients are applied together. No stale gradients.

In practice, A2C tends to match or exceed A3C performance on modern hardware where parallel CPU/GPU computation is available. The synchronous nature makes it more reproducible and easier to debug.

---

## LECTURE 24: PPO, DDPG, and Model-Based RL

---

### 29. The Problem with Policy Gradient Step Size

Recall the policy gradient update:

    θ_{k+1} = θ_k + α · ∇_θ J_θ

Choosing the step size α is critically important and notoriously difficult in RL.

**In supervised learning**: if you take a step that is too large, the next batch of data will correct you. Your training data is fixed and independent of your parameters.

**In RL**: if you take a step that is too large and end up with a terrible policy, the *next* mini-batch of experience will be collected under that terrible policy. Bad data from a bad policy leads to another bad update, which makes the policy worse still. This is a catastrophic positive feedback loop. Unlike supervised learning, RL can spiral into failure if the step size is too large.

This is not just a theoretical concern — in practice, a policy gradient update that is too aggressive can permanently destroy a good policy.

We need a way to constrain the policy update so that the new policy is not too different from the old one. This is the motivation for Proximal Policy Optimization (PPO).

---

### 30. From Policy Gradient to PPO

PPO starts with the policy gradient objective and re-derives it in a way that naturally allows for a trust region constraint.

**Step 1**: The policy gradient objective with advantage:

    ∇J(θ) = E_{(s,a) ~ π_θ}[∇ log π_θ(a|s) · A(s, a)]

**Step 2**: Rewrite using importance sampling. We want to update the policy from θ_old to θ, but we collected data under π_{θ_old}. Using the importance sampling formula E_{x~p}[f(x)] = E_{x~q}[f(x) · p(x)/q(x)]:

    ∇J(θ) = E_{(s,a) ~ π_{θ_old}}[π_θ(s,a) / π_{θ_old}(s,a) · ∇ log π_θ(a|s) · A(s, a)]

**Step 3**: Define the probability ratio:

    r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

Note that r(θ_old) = 1. This ratio tells us: relative to the old policy, how much more or less likely is the new policy to take action a_t in state s_t?

**Step 4**: The surrogate objective becomes:

    J(θ) = E_{(s,a) ~ π_{θ_old}}[r_t(θ) · Â_t]

where Â_t is the estimated advantage.

**Step 5**: PPO clips this ratio to prevent large policy changes:

    L^CLIP(θ) = E_t[min(r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t)]

Let's understand the clipping:
- If Â_t > 0 (action was better than average), we want to increase r_t(θ), i.e., make the new policy more likely to take this action. But we cap at 1+ε, preventing too large an increase.
- If Â_t < 0 (action was worse than average), we want to decrease r_t(θ). But we cap at 1−ε, preventing too large a decrease.
- The min ensures we take the pessimistic (conservative) estimate.

Typical value: ε = 0.2. This means the new policy cannot be more than 20% more or less probable than the old policy for any given action.

**Why is this better than just constraining the step size α?**

Because the clipping operates on the probability ratio, which directly measures how different the policies are. A fixed step size in parameter space does not directly control how much the policy distribution changes (a small parameter change can cause a large policy change, depending on the parameterization). PPO's clipping directly controls the policy change, regardless of the parameterization.

**Practical algorithm**:
1. Collect trajectories using current policy π_{θ_old}.
2. Compute advantages Â_t using GAE (Generalized Advantage Estimation) or simpler methods.
3. Update θ by maximizing L^CLIP using multiple epochs of mini-batch gradient ascent.
4. Set θ_old ← θ and repeat.

The "multiple epochs" part is important: because we have the clipping, we can safely reuse the same trajectory data for multiple gradient steps, unlike standard policy gradient (where using data multiple times would give biased gradient estimates).

---

### 31. Deep Deterministic Policy Gradient (DDPG)

DDPG is designed specifically for **continuous action spaces** where actions are real-valued vectors (e.g., joint torques, steering angles).

In Q-learning, the policy improvement step requires:

    a* = argmax_a Q(s, a)

For discrete actions, this is easy — just evaluate Q for each action. For continuous actions, this requires solving a continuous optimization problem at every time step, which is computationally expensive.

DDPG's key insight: use a **deterministic policy network** μ_θ(s) that directly outputs an action (not a distribution over actions). Then, gradient of Q with respect to action can be backpropagated through the policy network.

**DDPG Algorithm**:

For each iteration:

1. **Roll-outs**: Execute current policy μ_θ(s) plus Gaussian or Ornstein-Uhlenbeck noise (for exploration, since the policy is deterministic). Collect transitions.

2. **Q-function (critic) update**:
   
   Compute gradient: g ∝ ∇_w Σ_t (Q_w(s_t, u_t) − Q̂(s_t, u_t))²
   
   where Q̂(s_t, u_t) = r_t + γ · Q_{w^-}(s_{t+1}, μ_{θ^-}(s_{t+1}))
   
   Here w^- and θ^- are target network parameters (frozen), and u_t is the action taken. This is the same target network trick as DQN.

3. **Policy (actor) update**:
   
   Compute gradient: g ∝ Σ_t ∇_θ Q_w(s_t, μ_θ(s_t, v_t))
   
   This gradient flows through Q_w with respect to its action input, and then through μ_θ with respect to θ (chain rule / backpropagation through the Q-network into the policy network).

The actor update is saying: to improve the policy, find the direction in parameter space that, if we change θ in that direction, would maximize Q. This is possible because both Q and μ are differentiable neural networks.

DDPG uses:
- Replay buffer (same as DQN)
- Target networks for both actor and critic (same idea as DQN)
- Ornstein-Uhlenbeck noise for temporally correlated exploration

DDPG is **off-policy** (uses replay buffer), which makes it sample-efficient. But it can be unstable and sensitive to hyperparameters.

---

### 32. Comparison: Actor-Critic vs. DDPG vs. PPO

| Aspect | Actor-Critic | DDPG | PPO |
|---|---|---|---|
| Type | Framework | Specific Algorithm | Specific Algorithm |
| Policy type | Either stochastic or deterministic | Mostly deterministic | Stochastic |
| On/off policy | Either | Off-policy | On-policy |
| Action space | Any | Continuous | Mostly continuous or discrete |
| Stability | Medium | Low to Medium | High |
| Sample efficiency | Medium | High (off-policy, replay) | Low to Medium (on-policy) |

Key takeaways:
- DDPG is very sample-efficient (reuses data via replay buffer) but unstable.
- PPO is stable and reliable but requires fresh data each iteration (on-policy, less efficient).
- Actor-Critic is a general framework; DDPG and PPO are specific instantiations with particular design choices.

---

### 33. Model-Based Reinforcement Learning

Everything we have studied so far — Q-learning, SARSA, policy gradient, actor-critic — is **model-free**: the agent learns directly from experience without building an explicit model of the environment.

**Model-free**: Learn value function and/or policy directly from experience.

**Model-based**: First learn a *model* of the environment (transition dynamics and reward function), then use *planning* on that model to derive a value function or policy.

---

### 34. What Is a Model?

A model M is a representation of the MDP, parameterized by η:

    M = ⟨P_η, R_η⟩

where:
- P_η(s_{t+1}|s_t, a_t) ≈ P(s_{t+1}|s_t, a_t): the learned transition dynamics
- R_η(r_{t+1}|s_t, a_t) ≈ R(r_{t+1}|s_t, a_t): the learned reward function

We assume state space S and action space A are known. The model learns to predict the next state and reward.

Two ways to parameterize the reward model:

1. Expected reward: R^a_s = E[R_{t+1} | S_t = s, A_t = a] — just learn the mean reward.
2. Full distribution: R^a_{s,r} = Pr(R_{t+1} = r | S_t = s, A_t = a) — learn the full distribution of rewards.

---

### 35. Model Learning: It's Supervised Learning

Learning the model from experience {S_1, A_1, R_2, ..., S_T} is a supervised learning problem:

    S_1, A_1 → R_2, S_2
    S_2, A_2 → R_3, S_3
    ...
    S_{T-1}, A_{T-1} → R_T, S_T

We have input-output pairs where the input is (state, action) and the output is (next reward, next state).

- Learning s, a → r is a **regression problem** (predicting a real-valued reward).
- Learning s, a → s' is a **density estimation problem** (predicting a distribution over next states).

Examples of model types:
- **Table Lookup Model**: Explicitly store the MDP. Count visits and average.
- **Linear Expectation Model**: Linear function of state-action features.
- **Linear Gaussian Model**: Gaussian distribution with mean and variance linear in features.
- **Gaussian Process Model**: A full nonparametric Bayesian model.
- **Deep Belief Network Model**: Neural network model.

---

### 36. Table Lookup Model

The simplest model. Count the number of times each (state, action, next-state) triple was observed:

    P̂^a_{s,s'} = (1 / N(s,a)) · Σ_{t=1}^{T} 1(S_t, A_t, S_{t+1} = s, a, s')

This is just the empirical frequency of transitions. Similarly for rewards:

    R̂^a_s = (1 / N(s,a)) · Σ_{t=1}^{T} 1(S_t, A_t = s, a) · R_t

The indicator function 1(condition) is 1 if the condition is true, 0 otherwise.

Alternatively, you can just store all experience tuples ⟨S_t, A_t, R_{t+1}, S_{t+1}⟩ and, when sampling the model, randomly pick a stored tuple matching ⟨s, a, ·, ·⟩.

**Example** (from slides): Two states A and B, one action a, 8 episodes.

Experience collected: A,0,B,0 | B,1 | B,1 | B,1 | B,1 | B,1 | B,1 | B,0

From this data:
- Every time we were in A and took a, we moved to B → P(S_{t+1}=B | S_t=A, A_t=a) = 1
- From B, we always moved to terminal → P(S_{t+1}=T | S_t=B, A_t=a) = 1
- From A, reward was always 0 → R^a_A = 0
- From B, reward was 1 six times and 0 twice → R^a_B = 6/8 (if using expected reward model)

This is the table lookup model. Notice we have "constructed a table lookup model from experience."

---

### 37. Planning with the Model

Once we have a model M_η, we can solve the approximate MDP ⟨S, A, P_η, R_η⟩ using any planning algorithm:

- **Value Iteration**
- **Policy Iteration**
- **Tree search** (Monte Carlo Tree Search, etc.)

The key insight: DP methods (value iteration, policy iteration) require knowing the MDP. We cannot use them directly in RL because the true MDP is unknown. But if we learn a model M_η from data, we now have an approximate MDP that we can solve with DP. This is exactly what model-based RL does.

---

### 38. Sample-Based Planning

A simple and powerful approach: instead of solving the model analytically, use the model to *generate samples*, and then apply model-free RL to those samples.

    S_{t+1} ~ P_η(S_{t+1} | S_t, A_t)
    R_{t+1} = R_η(R_{t+1} | S_t, A_t)

Then apply Monte Carlo, SARSA, Q-learning, etc., to these *simulated* transitions. This is powerful because:
1. We can generate as many simulated transitions as we want (cheap, if the model is cheap to evaluate).
2. Model-free algorithms can be applied without modification.
3. Sample-based planning methods are often more computationally efficient than exact DP on the learned model.

---

### 39. Dyna Architecture: Integrating Learning and Planning

**Model-Free RL** (pure):
- No model
- Learn value function and/or policy from real experience only.

**Model-Based RL** (pure, sample-based planning):
- Learn a model from real experience
- Plan value function and/or policy from *simulated* experience only.

**Dyna**:
- Learn a model from real experience
- Learn and plan using *both real and simulated experience*.

Dyna is the integration of model learning and planning. The agent simultaneously:
1. Acts in the real environment, collecting real experience.
2. Learns the model from real experience (supervised learning).
3. Uses the model to simulate additional experience.
4. Updates the value function/policy from both real and simulated experience.

---

### 40. Tabular Dyna-Q Algorithm

Initialize Q(s, a) and Model(s, a) for all s ∈ S, a ∈ A(s).

Loop forever:
- (a) S ← current (nonterminal) state
- (b) A ← ε-greedy(S, Q)
- (c) Take action A; observe reward R and next state S'
- (d) Q(S, A) ← Q(S, A) + α[R + γ · max_a Q(S', a) − Q(S, A)]    ← direct RL update
- (e) Model(S, A) ← R, S'    ← model learning (deterministic assumption)
- (f) Loop n times (planning steps):
     - S ← random previously observed state
     - A ← random action previously taken in S
     - R, S' ← Model(S, A)
     - Q(S, A) ← Q(S, A) + α[R + γ · max_a Q(S', a) − Q(S, A)]    ← simulated experience update

Steps (d) is the direct RL update from real experience.
Step (e) updates the model.
Steps in the inner loop (f) are planning: we sample from the model and do additional Q-learning updates.

The inner loop repeats n times per real step. A larger n means more planning per unit of real experience. This dramatically speeds up learning when n is large — because each real transition gives us n additional "virtual" training updates.

**Results** (Figure 8.2): On a maze task, Dyna-Q with 50 planning steps reaches optimal performance in 14 episodes, while pure model-free RL (0 planning steps) takes hundreds of episodes. More planning steps → fewer real episodes needed to learn.

**Figure 8.3**: After just 2 episodes, Dyna-Q with n=50 has already filled in a good portion of the policy (many states pointing toward the goal), while Dyna-Q with n=0 has barely learned anything beyond the immediate path taken.

---

### 41. Dyna-Q+: Exploration Bonus

Dyna-Q has a weakness: if the environment changes (e.g., a previously blocked path becomes available), the learned model becomes inaccurate. Dyna-Q is slow to discover this because it preferentially replays frequently-visited state-action pairs and ignores long-untried ones.

**Dyna-Q+** adds an exploration bonus. For each state-action pair, it tracks τ(s, a) = the number of time steps since that pair was last tried in real experience.

The simulated reward is augmented:

    R + κ · √τ(s, a)

where κ is a small constant. The longer a pair has been untried, the higher the bonus. This encourages the agent to periodically revisit state-action pairs that have not been tried recently — discovering if the environment has changed.

**Results** (Figure 8.4): When the environment changes (a shortcut opens), Dyna-Q+ discovers and exploits the shortcut much faster than plain Dyna-Q.

---

### 42. What if the Model Is Inaccurate?

This is the fundamental limitation of model-based RL. If the learned model P_η, R_η does not accurately represent the true environment P, R, then the policy we compute is optimal for the *wrong* MDP. It may perform poorly in the real environment.

The performance of model-based RL is limited to:

    optimal policy for the approximate MDP ⟨S, A, P_η, R_η⟩

If the model is completely wrong, model-based RL can produce a very bad policy with high confidence (it is confidently wrong).

Solutions:
1. Use model-free RL when the model is wrong (detect model errors and fall back).
2. Maintain uncertainty in the model (Bayesian approaches, Gaussian Processes) and plan conservatively under uncertainty.
3. Use Dyna-style integration: use real experience as well as simulated experience, so model errors are partially corrected by real interactions.

---

## SUMMARY AND CONCEPTUAL MAP

Here is how all the concepts connect:

**Value-Based (Tabular → Function Approximation → Deep)**:
Tabular Q-learning → Linear FA with semi-gradient → DQN (nonlinear FA, replay, frozen target)

**Policy-Based**:
REINFORCE (MC, unbiased, high variance) → REINFORCE with baseline (reduced variance) → Policy Gradient Theorem (principled derivation) → Actor-Critic (bootstrapped critic, less variance, some bias) → A3C/A2C (parallel, stable)

**Advanced Algorithms**:
PPO (stable policy gradient with clipping) → DDPG (off-policy, continuous actions, deterministic policy)

**Model-Based**:
Table Lookup Model → Sample-Based Planning → Dyna (real + simulated experience) → Dyna-Q+ (exploration bonus for changing environments)

**Key Trade-offs to Understand**:
1. Bias vs. Variance: Monte Carlo (unbiased, high variance) vs. TD bootstrapping (biased, low variance)
2. On-policy vs. Off-policy: On-policy (PPO, A3C) is simpler and more stable; off-policy (DDPG, DQN) is more sample-efficient
3. Model-free vs. Model-based: Model-free is robust; model-based is sample-efficient but fails if model is inaccurate
4. Sample efficiency vs. Stability: DDPG is efficient but unstable; PPO is stable but less efficient

---

*End of Notes: Lectures 20–24*
