# Proof that the certificate bounds the Wilson Hessian (d = 4)

This file proves the step that the exact and interval checks of the certificate take as given: that positive
definiteness of the 96 plaquette matrices $M_p(\varphi)$, together with the finite checks on $G_y$ and $a_{p,r}$,
implies

$$|\mathrm{Hess}\,S_0(v,v)|\ \le\ h\,|v|^2,\qquad h=\tfrac{14420001}{10^6},$$

for every $\mathrm{SU}(N)$, $N\ge 2$, every finite box of $\mathbb Z^4$, every field, and every tangent vector
supported on free links. It is Lemma 5.1 of the 3D paper (romanov360/yang-mills-3d-forest-gauge, `lem:cert`) with
6 ends per site replaced by 8. Nothing else in the argument depends on the dimension. `validate4d.py` checks every
identity below numerically (`logs/validate4d.out`), but the proof does not use it.

## 0. Conventions

- **Metric.** $\langle X,Y\rangle=-\mathrm{Tr}(XY)=\mathrm{Re}\,\mathrm{Tr}(X^*Y)$ on $\mathfrak u(N)$. It is
  invariant under $\mathrm{Ad}(U)$ for every unitary $U$. In the code this is `ip`.
- **Action.** $S_0=\sum_p\mathrm{Re}\,\mathrm{Tr}\,Q_p$, with one term per unoriented plaquette. For
  $x\in\mathbb Z^4$ and $\mu<\nu$, the plaquette $p=(x;\mu,\nu)$ has links $e_1=(x,\mu)$, $e_2=(x+e_\mu,\nu)$,
  $e_3=(x+e_\nu,\mu)$, $e_4=(x,\nu)$ and holonomy $Q_p=Q_1Q_2Q_3^*Q_4^*$.
- **Tangent vectors.** $v=(X_e)_e$ with $X_e\in\mathfrak{su}(N)$, $|v|^2=\sum_e|X_e|^2$. Since the metric is
  bi-invariant, $t\mapsto e^{tX_e}Q_e$ is a geodesic, so
  $\mathrm{Hess}\,S_0(v,v)=\frac{d^2}{dt^2}S_0\big((e^{tX_e}Q_e)_e\big)\big|_{t=0}$.
- **Forest.** $F$ is the period-2 set of fixed links in `forest4d.py`. Each site $x$ is the head of exactly one link
  of $F$, namely $(x-e_{\mu(x)},\mu(x))$, where $\mu(x)$ depends on $x \bmod 2$.
  - *$F$ has no cycles.* In a cycle of $F$ with $k$ vertices and $k$ edges, each vertex is the head of at most one
    edge, so the cycle is directed. Each step of a directed path back along $F$ lowers $\sum_ix_i$ by one, so a
    directed path cannot close.
  - *Tangent vectors.* $X_e=0$ for $e\in F$. The lemma below does not use $Q_e=I$ on $F$, only $X_e=0$ there.
- **Box.** $\Lambda=\prod_i[m_i,n_i]\cap\mathbb Z^4$. A link lies in $\Lambda$ if both of its endpoints do, and a
  plaquette lies in $\Lambda$ if all four of its links do. The box action is $S_0^\Lambda=\sum_{p\subset\Lambda}\mathrm{Re}\,\mathrm{Tr}\,Q_p$.

## 1. Plaquette forms

Perturb $Q_i\mapsto e^{tX_i}Q_i$. Two rewrites move the exponentials next to each other:

- $Q_1e^{tX_2}=e^{t\,\mathrm{Ad}(Q_1)X_2}Q_1$;
- $e^{-tX_3}Q_4^*=Q_4^*\,e^{-t\,\mathrm{Ad}(Q_4)X_3}$.

Together they give

$$Q_p(t)=e^{tY_1}e^{tY_2}\,W\,e^{-tY_3}e^{-tY_4},\qquad Y_1=X_1,\ Y_2=\mathrm{Ad}(Q_1)X_2,\ Y_3=\mathrm{Ad}(Q_4)X_3,\ Y_4=X_4,\ W=Q_p .$$

Differentiating twice at $t=0$ and using cyclicity of the trace gives

$$f_p(Y,W)=\sum_{i}\mathrm{Re}\,\mathrm{Tr}(Y_i^2W)+2\,\mathrm{Re}\,\mathrm{Tr}(Y_1Y_2W)+2\,\mathrm{Re}\,\mathrm{Tr}(Y_3Y_4W)
-2\sum_{i\in\{1,2\},\,j\in\{3,4\}}\mathrm{Re}\,\mathrm{Tr}(Y_iWY_j),$$

which is `fp` in `certificate3d.py` and `cert_interval4d.py`. So, for any box,
$\mathrm{Hess}\,S_0^\Lambda(v,v)=\sum_{p\subset\Lambda}f_p$. The form $f_p$ is real-linear in $W$, so

$$f_p(Y,-W)=-f_p(Y,W). \tag{1.1}$$

Since $\mathrm{Ad}$ is an isometry, $|Y_r|=|X_{e_r}|$.

## 2. Ends and corners

At a site $y$, the eight **ends** are

$$o_\mu(y)=X_{(y,\mu)},\qquad i_\mu(y)=\mathrm{Ad}(Q^*_{(y-e_\mu,\mu)})\,X_{(y-e_\mu,\mu)},\qquad \mu=0,\dots,3 .$$

The end of a link outside $\Lambda$, or of a link in $F$, is $0$. Each link $e=(y,\mu)$ has two ends: $o_\mu$ at its
tail $y$ and $i_\mu$ at its head $y+e_\mu$. Both have norm $|X_e|$. In the code, $o_\mu$ is index $\mu$ and
$i_\mu$ is index $4+\mu$.

**Corner identities.** For $p=(x;\mu,\nu)$, Ad-invariance of $\langle\cdot,\cdot\rangle$ gives:

| corner | identity | derivation | coefficient |
|---|---|---|---|
| $x$ | $\langle Y_1,Y_4\rangle=\langle o_\mu,o_\nu\rangle(x)$ | by definition | $k_{14}$ |
| $x+e_\mu$ | $\langle Y_1,Y_2\rangle=\langle i_\mu,o_\nu\rangle(x+e_\mu)$ | $\langle X_1,\mathrm{Ad}(Q_1)X_2\rangle=\langle \mathrm{Ad}(Q_1^*)X_1,X_2\rangle$ | $k_{12}$ |
| $x+e_\nu$ | $\langle Y_3,Y_4\rangle=\langle o_\mu,i_\nu\rangle(x+e_\nu)$ | $\langle \mathrm{Ad}(Q_4)X_3,X_4\rangle=\langle X_3,\mathrm{Ad}(Q_4^*)X_4\rangle$ | $k_{34}$ |
| $x+e_\mu+e_\nu$ | $\langle Y_2,WY_3W^*\rangle=\langle i_\nu,i_\mu\rangle(x+e_\mu+e_\nu)$ | below | $k_{23}$ |

For the last row, $\mathrm{Ad}(W)\mathrm{Ad}(Q_4)=\mathrm{Ad}(Q_1Q_2Q_3^*)$, so

$$\langle\mathrm{Ad}(Q_1)X_2,\mathrm{Ad}(Q_1Q_2Q_3^*)X_3\rangle=\langle \mathrm{Ad}(Q_2^*)X_2,\mathrm{Ad}(Q_3^*)X_3\rangle .$$

In `certify4d.plaquettes` the corners are listed as `(x,mu,nu)`, `(x+mu, 4+mu, nu)`, `(x+mu+nu, 4+nu, 4+mu)` and
`(x+nu, mu, 4+nu)`. Their coefficients $k=2G_{\text{corner}}[a,b]$ multiply `C14`, `C12`, `C23` and `C34` in
`plaq_matrix`. The corner form of $p$ is

$$c_p(Y,W)=k_{14}\langle Y_1,Y_4\rangle+k_{12}\langle Y_1,Y_2\rangle+k_{34}\langle Y_3,Y_4\rangle+k_{23}\langle Y_2,WY_3W^*\rangle .$$

It is unchanged under $W\mapsto -W$.

**Each perpendicular pair occurs exactly once.** At a site $y$, take two ends in different directions
$\mu<\nu$. Each such pair is a corner of exactly one plaquette:

| pair | plaquette | corner |
|---|---|---|
| $(o_\mu,o_\nu)$ | $(y;\mu,\nu)$ | $x$ |
| $(i_\mu,o_\nu)$ | $(y-e_\mu;\mu,\nu)$ | $x+e_\mu$ |
| $(o_\mu,i_\nu)$ | $(y-e_\nu;\mu,\nu)$ | $x+e_\nu$ |
| $(i_\mu,i_\nu)$ | $(y-e_\mu-e_\nu;\mu,\nu)$ | $x+e_\mu+e_\nu$ |

This gives $4\binom42=24$ pairs per site, matching the $6\times4=24$ corners of the plaquettes per site. The only
other pairs of distinct ends at $y$ are the parallel pairs $(o_\mu,i_\mu)$. The certificate sets $G_y[o_\mu,i_\mu]=0$
for them.

**In a box.** If both links of a corner pair lie in $\Lambda$, then the three sites $y$, $y\pm e_\mu$ and
$y\pm e_\nu$ lie in $\Lambda$. Because $\Lambda$ is a product of intervals, the fourth corner of the plaquette lies
in $\Lambda$ too, so the plaquette lies in $\Lambda$. Conversely, a corner pair with a link outside $\Lambda$
contributes $0$.

**Site identity.** Sum over the sites of $\Lambda$, with the symmetric matrices $G_y$ taken from the period cell
($y \bmod 2$). Expanding $\sum_{a,b}=\sum_a+2\sum_{a<b}$ gives

$$\sum_{y}\sum_{a,b}G_y[a,b]\langle \mathrm{end}_a,\mathrm{end}_b\rangle(y)
=\sum_{p\subset\Lambda}c_p(Y,W)+\sum_{e}\big(G_{\mathrm{tail}(e)}[o_\mu,o_\mu]+G_{\mathrm{head}(e)}[i_\mu,i_\mu]\big)|X_e|^2 . \tag{2.1}$$

The off-diagonal terms are the corners. The parallel pairs have coefficient $0$, and the diagonal terms are the norms
of the ends. Each $G_y$ is positive semidefinite, and the Gram matrix of the ends at $y$ is positive semidefinite.
The trace of a product of two positive semidefinite matrices is $\ge0$, so the left side of (2.1) is $\ge0$.

## 3. The lemma

**Lemma.** Suppose that:

- every $G_y$ is real symmetric positive semidefinite with $G_y[o_\mu,i_\mu]=0$;
- $a_{p,r}\ge0$ for the free positions $r$ of each plaquette;
- for every $W\in\mathrm U(N)$, and every $Y\in\mathfrak u(N)^4$ with $Y_r=0$ at the fixed position,

$$f_p(Y,W)+c_p(Y,W)\ \le\ \sum_{r\ \text{free}}a_{p,r}|Y_r|^2 . \tag{3.1}$$

Then for every box $\Lambda$, every field $Q$ and every $v$ with $X=0$ on $F$ and off $\Lambda$,
$|\mathrm{Hess}\,S_0^\Lambda(v,v)|\le h|v|^2$, where

$$h=\max_{e\notin F}\Big(\sum_{p\ni e}a_{p,e}+G_{\mathrm{tail}(e)}[o_\mu,o_\mu]+G_{\mathrm{head}(e)}[i_\mu,i_\mu]\Big).$$

**Proof.**

1. **Upper bound.** Sum (3.1) over $p\subset\Lambda$. Plaquettes outside $\Lambda$ only add nonnegative weights, so
   dropping them leaves each link's total at most the right side of the definition of $h$. Adding (2.1), whose left
   side is $\ge0$, gives
   $$\mathrm{Hess}\,S_0^\Lambda(v,v)=\sum_pf_p\ \le\ \sum_p\Big(\sum_ra_{p,r}|Y_r|^2-c_p\Big)
   \ \le\ \sum_e\Big(\sum_{p\ni e}a_{p,e}+G_{\mathrm{tail}}[o,o]+G_{\mathrm{head}}[i,i]\Big)|X_e|^2\ \le\ h|v|^2 .$$
2. **Lower bound.** Apply (3.1) at $-W$, which lies in $\mathrm U(N)$. This replaces $f_p$ by $-f_p$ by (1.1) and
   leaves $c_p$ unchanged. The same chain then gives $-\mathrm{Hess}\,S_0^\Lambda(v,v)\le h|v|^2$. $\square$

Hypothesis (3.1) has to be stated for $\mathrm U(N)$, not for $\mathrm{SU}(N)$: for odd $N$, $-W\notin\mathrm{SU}(N)$.
Restricting to $Y\in\mathfrak{su}(N)$ and $W\in\mathrm{SU}(N)$ gives the $\mathrm{SU}(N)$ statement.

## 4. Reduction to U(2) phases

All three quantities in (3.1) are unchanged when every $Y_i$ and $W$ are conjugated by one unitary: $f_p$, $c_p$ and
$|Y_r|^2$. So we may take $W=\mathrm{diag}(w_1,\dots,w_N)$.

**Every term has the same shape.** Each term of $f_p$, of $c_p$ and of $|Y_r|^2$ has the form
$\mathrm{Re}\,\mathrm{Tr}(Y_iD_1Y_jD_2)$ with $D_1,D_2\in\{I,W,W^*\}$ diagonal:

- $\mathrm{Tr}(Y^2W)$: $D_1=I$, $D_2=W$;
- $\mathrm{Tr}(Y_iWY_j)$: $D_1=W$, $D_2=I$;
- $\langle Y_2,WY_3W^*\rangle$: $D_1=W$, $D_2=W^*$, with sign $-1$;
- $\langle Y_i,Y_j\rangle$: $D_1=D_2=I$, with sign $-1$.

Such a term equals $\mathrm{Re}\sum_{a,b}(Y_i)_{ab}(Y_j)_{ba}\,\phi(w_a,w_b)$.

**Splitting.** So $\Phi=\sum_ra_{p,r}|Y_r|^2-f_p-c_p$ is a sum of pieces:

- for each $a$, a piece in the diagonal entries $(Y_i)_{aa}$;
- for each pair $a<b$, a piece in the entries $(Y_i)_{ab}$ and $(Y_i)_{ba}$.

**Each piece is a U(2) case.** The $\{a,b\}$ piece is the $\mathrm U(2)$ form at $W'=\mathrm{diag}(w_a,w_b)$,
evaluated on $Y'_i=\begin{pmatrix}0&(Y_i)_{ab}\\(Y_i)_{ba}&0\end{pmatrix}\in\mathfrak u(2)$. The $\{a\}$ piece is the
same form evaluated on $Y'_i=\mathrm{diag}((Y_i)_{aa},0)$. In both cases $Y'_r=0$ at the fixed position.

So (3.1) for $\mathrm U(2)$ at all $W=\mathrm{diag}(e^{i\varphi_1},e^{i\varphi_2})$ implies (3.1) for every
$\mathrm U(N)$.

**The matrix.** In the orthonormal basis $iI/\sqrt2,\ i\sigma_k/\sqrt2$ of $\mathfrak u(2)$, the $\mathrm U(2)$ case
says that a real symmetric matrix $M_p(\varphi)$ is positive semidefinite. Its rows and columns are indexed by the
three free positions times four basis directions, so it is $12\times12$, because every plaquette of $F$ has exactly
one fixed link (`forest4d.py`). Its entries are affine in $\cos\varphi_a$, $\sin\varphi_a$,
$\cos(\varphi_1-\varphi_2)$ and $\sin(\varphi_1-\varphi_2)$. `cert_interval4d.py` builds it exactly in sympy:
$D_p-\sum_tF_t\,\mathrm{trig}_t-(k_{14}C_{14}+k_{12}C_{12}+k_{34}C_{34})-k_{23}C_{23}(\varphi_1-\varphi_2)$.

## 5. Proof of the bound

The hypotheses of the lemma are exactly what the computer checks establish:

| hypothesis | check |
|---|---|
| $G_y$ symmetric, $G_y[o_\mu,i_\mu]=0$, positive definite (exact $LDL^T$ pivots) | `cert_interval4d.py … finite` |
| $a_{p,r}\ge0$, carried only on the free positions | `cert_interval4d.py … finite` |
| maximum of the 48 per-link totals is exactly $14420001/10^6$ | `cert_interval4d.py … finite` |
| $M_p(\varphi)\succ0$ for all $\varphi\in[0,2\pi)^2$, all 96 plaquettes | `cert_interval4d.py … centre` (interval quadtree), logs `interval_h14.42_*.log` |

By periodicity, each plaquette and site of $\mathbb Z^4$ uses the data of its class mod 2. In the period cell, two
plaquettes can share two links, but this never matters: identities (2.1) and (3.1) are applied on $\mathbb Z^4$,
where the plaquettes are distinct.

By Sections 3 and 4, for every $\mathrm{SU}(N)$, every box and every field,

$$|\mathrm{Hess}\,S_0^\Lambda(v,v)|\ \le\ 14.420001\,|v|^2 .$$

Every result of BBL that uses their bound $29/2$ for this Hessian, in this gauge, therefore holds with $29/2$ replaced by
$14.420001$.

## 6. Independent checks

- **James Lockwood** re-ran the finite and all-phase checks independently: 48 distinct $M_p$, a $128\times128$ grid,
  and an explicit second-derivative interpolation bound.
- **`validate4d.py`** checks the identities of Sections 1–4 in floating point, on the torus of side 4:
  - $\sum_pf_p$ against an independent Hessian code (`fasthess`, SU(2)) and against finite differences (SU(3));
  - the site identity (2.1);
  - (3.1) at random $\mathrm U(3)$ and $\mathrm U(4)$ points.

  The largest discrepancies are $4\times10^{-11}$ and $4\times10^{-15}$, and all slacks are positive.
