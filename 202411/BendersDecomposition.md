# Benders Decomposition for MILP

**Benders decomposition** is a *row-generation* technique used to solve large-scale problems by dividing them into a **master problem** and a **subproblem**.

## Mixed-Integer Linear Programming (MILP)

Consider a mixed-integer linear programming (MILP) problem in the following form:

$$
\begin{aligned}
&\operatorname*{minimize}_{\bm x, \bm y}
&&  \bm c_x^\top \bm x + \bm c_y^\top \bm y, \\
&\operatorname{subject~to}
&&  \bm A_x \bm x + \bm A_y \bm y \leq \bm b, \\
&&& \bm x \in \mathbb{Z}^n, \\
&&& \bm y \in \mathbb{R}_{\ge 0}^m,
\end{aligned}
\tag{P}
$$

where

- $\bm x \in \mathbb{Z}^n$: integer variables
- $\bm y \in \mathbb{R}_{\ge 0}^m$: continuous variables
- $\bm c_x \in \mathbb{R}^n$, $\bm c_y \in \mathbb{R}^m$: objective coefficients
- $\bm A_x \in \mathbb{R}^{p \times n}$, $\bm A_y \in \mathbb{R}^{p \times m}$: constraint coefficients
- $\bm b \in \mathbb{R}^p$: right-hand side of constraints
-  inequalities hold element-wise

## Decomposing into Two-Stage Optimization

First we divide the original problem into a two-stage optimization problem by introducing an auxiliary variable $\phi \in \mathbb{R}$ as:

$$
\begin{aligned}
&\operatorname*{minimize}_{\bm x, \phi}
&&  \bm c_x^\top \bm x + \phi, \\
&\operatorname{subject~to}
&&  \bm A_x \bm x \leq \bm b, \\
&&& \phi \ge \bm c_y^\top \bm y^\ast (\bm x), \\
&&& \bm x \in \mathbb{Z}^n, \\
&&& \phi \in \mathbb{R},
\end{aligned}
\tag{M}
$$

where $\bm y^\ast(\bm x)$ is the optimal solution to the following linear programming:

$$
\begin{aligned}
\bm y^\ast(\bm x) &\coloneqq
\arg
\left\{
    \begin{aligned}
    & \operatorname*{minimize}_{\bm y}
    &&  \bm c_y^\top \bm y, \\
    &\operatorname{subject~to}
    &&  \bm A_x \bm x + \bm A_y \bm y \leq \bm b, \\
    &&& \bm y \in \mathbb{R}_{\ge 0}^m.
    \end{aligned}
\right.
\end{aligned}
\tag{S}
$$

The first stage problem $\text{(M)}$ is called the **master problem** and the second stage problem $\text{(S)}$ is called the **subproblem**. Note that this two-stage optimization problem is equivalent to the original problem $\text{(P)}$.

<blockquote>

**Proposition 1**

The optimal solution to the original problem $\text{(P)}$ is the same as the optimal solution to the two-stage optimization problem $\text{(M)}$-$\text{(S)}$.

</blockquote>

## Assumption of Strong Duality

Now we consider the subproblem $\text{(S)}$ defined as:

$$
\begin{aligned}
    & \operatorname*{minimize}_{\bm y}
    &&  \bm c_y^\top \bm y, \\
    &\operatorname{subject~to}
    &&  \bm A_y \bm y \leq \bm b - \bm A_x \bm x, \\
    &&& \bm y \in \mathbb{R}_{\ge 0}^m,
\end{aligned}
\tag{S}
$$

and its dual problem $\text{(S-D)}$ defined as:

$$
\begin{aligned}
    &\operatorname*{maximize}_{\bm u}
    &&  \bm u^\top (\bm b - \bm A_x \bm x), \\
    &\operatorname{subject~to}
    &&  \bm u^\top \bm A_y \leq \bm c_y^\top, \\
    &&& \bm u \in \mathbb{R}_{\ge 0}^p.
\end{aligned}
\tag{S-D}
$$

The following theorem allows further reformulation of the two-stage optimization problem $\text{(M)}$-$\text{(S)}$.

<blockquote>

**Theorem 1 (Strong Duality Theorem)**

If the dual problem $\text{(S-D)}$ has an optimal solution $\bm u^\ast(\bm x)$ for given $\bm x$, then the primal problem $\text{(S)}$ also has an optimal solution $\bm y^\ast(\bm x)$ for given $\bm x$ and the following equality holds:

$$
\begin{aligned}
\bm c_y^\top \bm y^\ast(\bm x) = \bm u^\ast (\bm x)^\top (\bm b - \bm A_x \bm x).
\end{aligned}
$$

</blockquote>

Thus, by assuming $\text{(S-D)}$ has an optimal solution, the two-stage problem can be rewritten as:

$$
\begin{aligned}
&\operatorname*{minimize}_{\bm x, \phi}
&&  \bm c_x^\top \bm x + \phi, \\
&\operatorname{subject~to}
&&  \bm A_x \bm x \leq \bm b, \\
&&& \phi \ge \bm u^\ast(\bm x)^\top (\bm b - \bm A_x \bm x), \\
&&& \bm x \in \mathbb{Z}^n, \\
&&& \phi \in \mathbb{R},
\end{aligned}
\tag{M'}
$$

where

$$
\begin{aligned}
\bm u^\ast(\bm x) &\coloneqq
\arg
\left\{
    \begin{aligned}
    &\operatorname*{maximize}_{\bm u}
    &&  \bm u^\top (\bm b - \bm A_x \bm x), \\
    &\operatorname{subject~to}
    &&  \bm u^\top \bm A_y \leq \bm c_y^\top, \\
    &&& \bm u \in \mathbb{R}_{\ge 0}^p.
    \end{aligned}
\right.
\end{aligned}
\tag{S-D}
$$

Or equivalently, we can rewrite the two-stage optimization problem as:

$$
\begin{aligned}
&\operatorname*{minimize}_{\bm x, \phi}
&&  \bm c_x^\top \bm x + \phi, \\
&\operatorname{subject~to}
&&  \bm A_x \bm x \leq \bm b, \\
&&& \phi \ge \bm u^\top (\bm b - \bm A_x \bm x),
&& \forall \bm u \in \left\{
    \bm u 
    ~\middle|~
        \begin{aligned}
            & \bm u^\top \bm A_y \leq \bm c_y^\top, \\
            & \bm u \in \mathbb{R}_{\ge 0}^p
        \end{aligned}
    \right\}, \\
&&& \bm x \in \mathbb{Z}^n, \\
&&& \phi \in \mathbb{R},
\end{aligned}
\tag{M''}
$$

<blockquote>

**Proposition 2**

If the dual problem $\text{(S-D)}$ has an optimal solution $\bm u^\ast(\bm x)$ for any feasible $\bm x$, then the optimal objective value of the original problem $\text{(P)}$ is the same as the optimal objective value of the optimization problem $\text{(M'')}$.

</blockquote>

## Approximation via Benders Cuts

The problem $\text{(M'')}$ has an infinite number of constraints

$$
\begin{aligned}
\phi \ge \bm u^\top (\bm b - \bm A_x \bm x),
&& \forall \bm u \in \left\{
    \bm u 
    ~\middle|~
        \begin{aligned}
            & \bm u^\top \bm A_y \leq \bm c_y^\top, \\
            & \bm u \in \mathbb{R}_{\ge 0}^p
        \end{aligned}
    \right\}
\end{aligned}
$$

and is practically intractable. To mitigate this issue, Benders decomposition approximates the infinite number of constraints with a finite number of constraints,

$$
\begin{aligned}
\phi \ge \bm u^\top (\bm b - \bm A_x \bm x),
&& \forall \bm u \in \{ \text{already found $\bm u$'s} \}.
\end{aligned}
$$

The Benders decomposition algorithm is as follows:

1. Let $k = 1$.
2. **(MASTER PROBLEM)**
    Solve the relaxed master problem defined as
    $$
    \begin{aligned}
    &\operatorname*{minimize}_{\bm x, \phi}
    &&  \bm c_x^\top \bm x + \phi, \\
    &\operatorname{subject~to}
    &&  \bm A_x \bm x \leq \bm b, \\
    &&& \phi \ge \bm u^{k\top} (\bm b - \bm A_x \bm x),
    && \forall k = 1, 2, \ldots, K, \\
    &&& \bm x \in \mathbb{Z}^n, \\
    &&& \phi \in \mathbb{R},
    \end{aligned}
    $$
    and save the optimal solution to $\bm x^k$ and $\phi^k$.
3. **(SUBPROBLEM)**
    Solve the dual of subproblem for the current $\bm x_k$ defined as
    $$
    \begin{aligned}
    &\operatorname*{maximize}_{\bm u}
    &&  \bm u^\top (\bm b - \bm A_x \bm x_k), \\
    &\operatorname{subject~to}
    &&  \bm u^\top \bm A_y \leq \bm c_y^\top, \\
    &&& \bm u \in \mathbb{R}_{\ge 0}^p,
    \end{aligned}
    $$
    and save the optimal solution to $\bm u^k$.
4. **(SATISFACTION CHECK)**
    If the constraint $\bm \phi^k \ge \bm u^{k\top} (\bm b - \bm A_x \bm x^k)$ is satisfied, go to step 5. Otherwise, add the following constraint to the master problem:
    $$
    \begin{aligned}
    \phi \ge \bm u^{k\top} (\bm b - \bm A_x \bm x),
    \end{aligned}
    $$
    and go to step 2 with $k = k + 1$.
5. As a final step, solve the primal subproblem for the current $\bm x_k$ defined as
    $$
    \begin{aligned}
    \bm y^k &\coloneqq
    \arg
    \left\{
        \begin{aligned}
        & \operatorname*{minimize}_{\bm y}
        &&  \bm c_y^\top \bm y, \\
        &\operatorname{subject~to}
        &&  \bm A_x \bm x_k + \bm A_y \bm y \leq \bm b, \\
        &&& \bm y \in \mathbb{R}_{\ge 0}^m.
        \end{aligned}
    \right.
    \end{aligned}
    $$
    and terminate the algorithm.

The constraint

$$
\begin{aligned}
\phi \ge \bm u^{k\top} (\bm b - \bm A_x \bm x),
\end{aligned}
$$

is called a **Benders cut**.

<blockquote>

**Theorem 2 (Weak Duality Theorem)**

If $\bm y$ is a feasible solution to the subproblem, and $\bm u$ is a feasible solution to the dual of the subproblem, then the following inequality holds:

$$
\begin{aligned}
\bm c_y^\top \bm y \ge \bm u^\top (\bm b - \bm A_x \bm x).
\end{aligned}
$$

</blockquote>

According this theorem, if both $\bm x^k$ and $\bm u^k$ satisfy this constraint, then the constraints

$$
\begin{aligned}
\phi^k \ge \bm u^\top (\bm b - \bm A_x \bm x^k),
&& \forall \bm u \in \left\{
    \bm u 
    ~\middle|~
        \begin{aligned}
            & \bm u^\top \bm A_y \leq \bm c_y^\top, \\
            & \bm u \in \mathbb{R}_{\ge 0}^p
        \end{aligned}
    \right\}
\end{aligned}
$$

are all satisfied. This means that the optimal solution to the master problem is the same as the optimal solution to the original problem.

Note that the algorithm above implicitly assumes that the dual of the subproblem always has an optimal solution. For the case where the subproblem is unbounded or infeasible, we need to add some modifications to the algorithm. Refer to [Benders (1962), Konno (1981), Wikipedia, etc.] for more details.

## References

- [Wikipedia] https://en.wikipedia.org/wiki/Benders_decomposition.
- [Benders (1962)] J. F. Benders: "Partitioning procedures for solving mixed-variables programming problems", Numerische Mathematik **4**, 238–252 (1962)
- [Konno (1981)] 今野浩「整数計画法」産業図書 (1981)
