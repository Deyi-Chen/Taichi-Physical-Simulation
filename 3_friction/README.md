## Friction Comparison

We compare two friction coefficients under the same setup.

Particularly, in the case of high friction, motion quickly damps out and reaches equilibrium. And in the case of low friction: object continues sliding down the slope.


| $\mu = 0.3$ (High friction) | $\mu = 0.01$ (Low friction) |
|-----------------------------|-----------------------------|
| ![](miu_0.3_friction.gif)   | ![](miu_0.01_friction.gif)  |


## Tangential Relative Velocity

We add friction to each contact pair $k$. Then, the tangential relative velocity is defined as:

```math
v_k = T_k(x)^T v \in \mathbb{R}^d
```

Specifically, the relative velocity is:
$$
v_k = (I_3 - n_k n_k^T)(v_p - v_s)
$$


As a result, define the global velocity:

$$
v = \begin{bmatrix}
v_p \\
v_s
\end{bmatrix} \in \mathbb{R}^6
$$

Then:

$$
T_k(x) =
\begin{bmatrix}
I_3 - n_k(x)n_k(x)^T \\
n_k(x)n_k(x)^T - I_3
\end{bmatrix}
\in \mathbb{R}^{6 \times 3}
$$



## Friction Energy

We define a smooth friction potential:

$$
P_f(x) = \sum_k \mu \lambda_k^n \, f_0\!\left( \|\bar v_k\| \, \hat h \right)
$$

where:

$$
\bar v_k = (T_k^n)^T v(x)
$$

- $\mu$: friction coefficient  
- $\lambda_k^n$: normal force magnitude (frozen at time step $n$)  
- $T_k^n$: tangential operator  
- $\hat h$: scaled time step  



## Smooth energy function

To avoid non-differentiability at $\|v\| = 0$, we define a smooth function
$f_0(y)$ as:

$$
f_0(y) =
\begin{cases}
-\dfrac{y^3}{3\epsilon_v^2 \hat h^2}
+ \dfrac{y^2}{\epsilon_v \hat h}
+ \dfrac{\epsilon_v \hat h}{3},
& y \in [0, \epsilon_v \hat h] \\[10pt]
y, & y \ge \epsilon_v \hat h
\end{cases}
$$

## Smooth Friction Scaling Function

The derivative of the smooth energy defines a scaling function:

$$
f_1(y) = f_0'(y)
$$

which acts as a smooth approximation of the Coulomb friction coefficient.

We define:

$$
f_1(y) =
\begin{cases}
-\dfrac{y^2}{\epsilon_v^2} + \dfrac{2y}{\epsilon_v},
& y \in [0, \epsilon_v] \\[10pt]
1, & y \ge \epsilon_v
\end{cases}
$$



## Gradient

Using the chain rule:

$$
\nabla_x P_f(x)
=
\sum_k \mu \lambda_k^n
\frac{\partial f_0(\|\bar v_k \hat h\|)}{\partial x}
$$

which expands to:

$$
\nabla_x P_f(x)
=
\sum_k \mu \lambda_k^n
\cdot
Df_0(\|\bar v_k \hat h\|)
\cdot
\frac{\bar v_k}{\|\bar v_k\|}
\cdot
(T_k^n)^T
\cdot
\frac{\partial v}{\partial x}
$$


## Hessian

The Hessian is obtained by differentiating the gradient:

$$
\nabla_x^2 P_f(x)
=
\sum_k \mu \lambda_k^n \, T_k^n \, H_k \, (T_k^n)^T \, \frac{\partial v}{\partial x}
$$

where $H_k$ is the Hessian with respect to the local velocity $\bar v_k$:

$$
H_k
=
f_1'(\|\bar v_k\|)
\frac{\bar v_k \bar v_k^T}{\|\bar v_k\|^2}
+
\frac{f_1(\|\bar v_k\|)}{\|\bar v_k\|}
\left(
I - \frac{\bar v_k \bar v_k^T}{\|\bar v_k\|^2}
\right)
$$



### Derivation Components

Let:

$$
g(v) = f_1(\|v\|)\frac{v}{\|v\|}
$$

Then:

$$
Dg(v)
=
f_1'(\|v\|)
\frac{v v^T}{\|v\|^2}
+
\frac{f_1(\|v\|)}{\|v\|}
\left(
I - \frac{v v^T}{\|v\|^2}
\right)
$$


