## Demo
![](contact.gif)

## Barrier Method

### Energy
In simulation, we may want to fix some points, namely 
```math
\min_x \; E(x) = \frac{1}{2} \left\| x - (x^n + h v^n) \right\|_M^2 + h^2 P(x)
\quad \text{s.t.} \quad Ax = b \quad \text{and} \quad \forall k,\; d_k(x) \ge 0
```

We convert the constraints using a barrier potential energy: 

$$
P_b(x) = \sum_{i,j} w_i \, b(d_{ij}(x))
$$

$$
b(d_{ij}(x)) =
\begin{cases}
\frac{\kappa}{2} \left( \frac{d_{ij}}{\hat{d}} - 1 \right) \ln \frac{d_{ij}}{\hat{d}}, & d_{ij} < \hat{d} \\
0, & d_{ij} \ge \hat{d}
\end{cases}
$$
As $d \to 0^+$, the barrier energy grows to infinity, 
preventing the solver from allowing interpenetration.

### Gradient (Force)

Applying the chain rule, and let $s_i = \frac{d_i}{\hat{d}}$

$$
\nabla P_b = \sum_{i,j} w_i \cdot \frac{\kappa}{2\hat{d}} 
\left( \ln s_{ij} + \frac{s_{ij} - 1}{s_{ij}} \right) \nabla d_{ij}(x)
$$

$$
f_{\text{contact}} = - \nabla P_b
$$

Here,
- $w_i$ is the volume weight
- $b'(d_i)$ controls the contact strength
- $\nabla d_i$ gives the contact normal direction

### Hessian of Barrier Energy

$$
\nabla  P_b = \sum_i w_i \left[ b''(d_i)\, n_i n_i^T + b'(d_i)\, \nabla^2  d_i \right]
$$

$$
\text{Hessian} = b''(d)\, nn^T + b'(d)\, D^2 d
$$

This expression naturally decomposes into two terms:

#### 1. Normal stiffness term (dominant)

$$
b''(d)\, nn^T
$$

This outer product matrix controls stiffness along the normal direction. 
A large value means that small changes in distance produce large changes in force magnitude, leading to a stiff response. 
A smaller value results in a softer interaction.

#### 2. Curvature term

$$
\nabla^2  d_i 
$$

It introduces curvature effects, namely how the direction of the force changes.

## Filtered line search 
This procedure ensures no collision during the process, and energy decrease via backtracking line search.
### Step 1: Compute Newton direction

$$
p = -H^{-1} g
$$


### Step 2: Compute maximum feasible step size (CCD)

$$
\alpha_c = \min_{j,k} \alpha_{jk}^C
$$


### Step 3: Initialize step size

$$
\alpha = \eta \alpha_c \quad (\eta \approx 0.9)
$$


### Step 4: Filtered line search

$$
\text{while } \exists (i,j): d_{ij}(x + \alpha p) < 0 
\;\; \text{or} \;\; E(x + \alpha p) > E(x):
\quad \alpha \leftarrow \frac{\alpha}{2}
$$


### Step 5: Update

$$
x \leftarrow x + \alpha p
$$

