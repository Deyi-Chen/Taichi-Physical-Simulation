## Demo
![](sticky_dirichlet.gif)
## Sticky Dirichlet Boundary Condition


In the previous section, we showed how the implicit mass–spring system can be formulated as an optimization problem. Here we consider the case where certain degrees of freedom are fixed, for example when part of a cloth is attached to a wall.

This leads to a constrained optimization problem where the energy to minimize is

```math
E(x) =
\frac{1}{2}(x-\tilde{x}^n)^T M (x-\tilde{x}^n)
+ \Delta t^2 P(x)
```

subject to

```math
Ax=b
```

where the constraint enforces fixed positions for certain DOFs.

---

### Newton Step

To minimize the energy, we solve the Newton system

```math
Hp = -g
```

where

```math
g = \nabla E(x), \qquad H = \nabla^2 E(x)
```

and update

```math
x^{k+1} = x^k + p
```

For a Dirichlet-constrained DOF $i$, the position must remain unchanged during the Newton step:

```math
x_i^{k+1} = x_i^k
```

which implies

```math
p_i = 0
```

---

### Sticky Dirichlet Modification

Instead of solving the full constrained KKT system, we enforce the constraint directly in the linear system.

For each constrained DOF $i$:

Set the gradient entry to zero

```math
g_i = 0
```

Modify the Hessian such that

```math
H_{ii} = 1
```

and

```math
H_{ij} = 0, \quad H_{ji} = 0 \quad (i \neq j)
```

This replaces the original equation

```math
\sum_j H_{ij} p_j = -g_i
```

with the constraint

```math
p_i = 0
```

Thus the constrained DOF becomes decoupled from the rest of the system, ensuring that its value remains fixed while the remaining DOFs are moving.

