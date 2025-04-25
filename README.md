# Bayesian Inverse Problems for PDEs with Discontinuous Coefficients  
### An Adaptive Finite Element Approach

This repository accompanies my Bachelor’s thesis, **“Bayesian Inverse Problems for Partial Differential Equations with Discontinuous Coefficients: An Adaptive Finite Element Approach.”**  
It contains the source code and reproducibility material for all theoretical and numerical results. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}  

---

## 1  Overview

Real-world materials often possess abrupt changes in conductivity, elasticity or diffusivity.  
Such jumps appear in elliptic PDEs with *discontinuous coefficients* and challenge both analysis and computation:

1. **Forward problem** – solve  
   \[
   -\nabla\!\cdot\!\bigl(a(x,r)\,\nabla u\bigr)=f,\qquad 
   u|_{\partial D}=0,\qquad 
   a(x,r)=\begin{cases}a_1,&x\notin B(x_0,r)\\ a_2,&x\in B(x_0,r)\end{cases}
   \]
2. **Inverse problem** – recover the unknown radius \(r\) of the inclusion \(B(x_0,r)\) from noisy observations \( \delta = G(u) + \eta\).

The thesis casts the inverse task in a **Bayesian** framework and shows that an **Adaptive Finite Element Method (AFEM)** yields near-optimal accuracy *without* manually fitting the mesh to the discontinuity.

---

## 2  Key Contributions
| Area | Contribution | Highlights |
|------|--------------|------------|
| **Well-posedness** | Proved continuity of the potential \(\Phi(r,\delta)\) and absolute continuity of the posterior \( \rho^\delta \) wrt. the prior | Guarantees a unique and stable Bayesian update |
| **AFEM theory** | Extended Bonito–DeVore–Nochetto (2013) results to the PDE with jump coefficients | Error in \(H^1\!\)-seminorm decays like \( \mathcal{O}(N^{-1/d}) \) |
| **Posterior error bound** | Derived \( d_{\text{Hell}}(\rho^\delta,\rho^\delta_{h}) \le C\,N^{-1/d} \) | Links mesh size directly to statistical accuracy |
| **Numerical study** | 1-D results shows AFEM beats uniform mesh in energy norm and Hellinger distance | Up to **2×** faster convergence for equal DoF |
| **MCMC integration** | Combined AFEM forward solver with Metropolis–Hastings | 10 000-sample chain recovers true \(r\) within 0.27 mean |


## 3  Repository Layout
├── Main Scripts
│   ├── Solver4.py [Solver4.py](Main Results) ** forward solver based on Degree of Freedoms.
│   └── Solver3.py ** forward solver based on Iteration Count. 
│   └── Log
│   └── Hellinger Distance 2.ipynb 
│   └── Bayesian Inverse
├── data
└── docs
