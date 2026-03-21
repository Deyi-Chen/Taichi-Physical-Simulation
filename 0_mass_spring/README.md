## Stability comparison under same perturbation

We apply the same initial perturbation to both explicit and implicit case, and explicit euler becomes unstable and diverges quickly. The system energy increases artificially, leading to numerical explosion (gray screen). On the other hand, implicit euler remains stable and produces physically plausible results.


| Explicit (diverges) | Implicit (stable) |
|--------------------|------------------|
| ![](explicit.gif)  | ![](implicit.gif) |

## Explicit time integration

```math
x^{n+1}=x^n+\Delta t\, v^n
```

```math
v^{n+1}=v^n+ \Delta t\, M^{-1} f^n
```


The explicit Euler is computationally cheap, meaning no linear system solve is required. However, its stability depends on the time step size, and the energy may increase over time for large time steps. 

## Implicit time integration 
```math
x^{n+1} = x^n + \Delta t \, v^{n+1}
```

```math
v^{n+1}=v^n+ \Delta t\, M^{-1}f^{n+1}
```

After some algebraic transformations, the implicit Euler can be also written as 
```math
g(x^{n+1}) =
M(x^{n+1}-(x^n+\Delta t \ v^n)) - Δt^2 f(x^{n+1}) = 0
```
The implicit Euler requires solving a linear system, but is more stable compared to the explicit version. 

## Newton method
To find a plausible solution to the implicit system, we can use Taylor's first order expansion on $g(x)$. Specifically, $g(x^i) + g'(x^i)(x - x^i) = 0$. As a result, the iteration formula is:
```math
x^{i+1} = x^i - (g'(x^i))^{-1} g(x^i)
```

## Optimization lens 
Observe the formula $M\left(x^{n+1} - (x^n + \Delta t\, v^n)\right)-\Delta t^2 f^{n+1}=0$. Let $x^{n+1} = x$ and define $\tilde{x}^n = x^n + \Delta t\, v^n$, then the equation becomes
$M(x - \tilde{x}^n) - \Delta t^2 f(x) = 0$ 

Since $M$ is a symmetric matrix, we know that 
```math
\nabla_x \left( \frac{1}{2} (x-a)^T M (x-a) \right) = M(x-a)
```

So the term $M(x-\tilde{x}^n)$ can actually be seen as the gradient of a quadratic energy.
On the other hand, if the force is conservative, we can write
```math
f(x) = -\nabla P(x)
```
where $P(x)$ is the potential energy.

This means the whole equation can be viewed as

```math
\nabla_x \left(
\frac{1}{2}(x-\tilde{x}^n)^T M (x-\tilde{x}^n)
+ \Delta t^2 P(x)
\right) = 0
```

As a result, solving the implicit Euler is equivalent to minizing an energy function. 

## Inertia term

Since M is a symmetric matrix, then the gradient of the first term 

```math
E_{\text{inertia}}(x)
=
\frac{1}{2}(x-\tilde{x}^n)^T M (x-\tilde{x}^n)
```

can be written as 
```math
\nabla E_{\text{inertia}}(x)
=
M(x-\tilde{x}^n)
```

and the Hessian is simply

```math
\nabla^2 E_{\text{inertia}}(x)
=
M
```


## Mass spring term
To avoid computing the square root, and considering the strain of the mass spring system, the second term can be written as

```math
P_e(x)=\frac12 k l^2
\left(
\frac{\|x_1-x_2\|^2}{l^2}-1
\right)^2
```

The gradient of the energy term is:
```math
2k
\left(
\frac{\|x_1-x_2\|^2}{l^2}-1
\right)
(x_1-x_2)
```

and the hessian of the mass spring term is: 
```math
\frac{2k}{l^2}
\left(
2(x_1-x_2)(x_1-x_2)^T
+
(\|x_1-x_2\|^2-l^2)I
\right)
```

Particularly, the full Hessian for the spring pair has the structure:

```math
\begin{bmatrix}
H & -H \\
-H & H
\end{bmatrix}
```

## Solver

To minimize the energy $E(x)$, we use Newton's method.

Using the second-order Taylor expansion around the current iterate $x^i$, we have

```math
E(x) \approx
E(x^i)
+
\nabla E(x^i)^T (x-x^i)
+
\frac12 (x-x^i)^T \nabla^2 E(x^i) (x-x^i)
```

To find the minimum of this quadratic approximation, we set its gradient to zero,

```math
\nabla E(x^i) + \nabla^2 E(x^i)(x-x^i) = 0
```

Let

```math
g = \nabla E(x^i), \qquad H = \nabla^2 E(x^i)
```

Then we obtain

```math
H(x-x^i) = -g
```

Denote the Newton step as

```math
p = x-x^i
```

which leads to the linear system

```math
Hp = -g
```

Finally, the update becomes

```math
x^{i+1} = x^i + p
```
