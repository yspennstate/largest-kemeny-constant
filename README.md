# The largest Kemeny constant and the largest relaxation time of a graph

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

For the simple random walk on a connected graph with $n$ vertices, let $\tau_0$ be Kemeny's constant and
$\tau_2=1/\mu_2$ the relaxation time (rate-one walk, $\mu_2$ the spectral gap of $I-P$). Aldous and Fill
(Reversible Markov Chains and Random Walks on Graphs, Open Problem 6.13) conjectured that the maxima of both
quantities over connected $n$-vertex graphs are asymptotic to $n^3/54$. The paper proves this:

- every connected graph on $n$ vertices has $\tau_2\le\tau_0\le n^3/54+\tfrac{37}{8}n^{5/2}$;
- the barbell made of two copies of $K_a$, $a=\lfloor n/3\rfloor$, joined by a path with $n-2a+1$ edges has
  $\tau_2\ge(n-2)(n-5)(n+3)/54$.

The optimal barbell and its Kemeny constant $n^3/54+O(n^2)$ are due to Faught, Kempton and Knudson
(arXiv:2108.01061); the upper bound uses the electrical formula for Kemeny's constant and a comparison with a tree.

## Code

Python 3 with numpy. `code/check.py` writes `code/check.out` next to itself. It computes Kemeny's constant of small
barbells exactly in rational arithmetic and compares it with the spectral and resistance formulas, compares the
barbell lower bound with $\tau_2$ and $\tau_0$ up to $n=600$, and checks every inequality in the proof of the upper
bound on 400 random connected graphs (Section 5 of the paper).

## Funding

The research presented in this paper was supported by the European Research Council (ERC) under the European
Union's Horizon Europe research and innovation programme (grant agreement No. 101041711), by the Simons Foundation
as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning, by Heights Labs, by
Convex Nexus Capital, by the Israel Science Foundation (grant number 2258/19), and by the Israel Science Foundation
(ISF Grant 4101/25). Computational resources were provided in part by Google Cloud through its education credits
program (credit 530550103).
