"""Checks for the bound max Kemeny ~ max relaxation time ~ n^3/54.

Part 1: exact Kemeny constants of small barbells (fundamental matrix, rational
        arithmetic) against the spectral and resistance formulas in floating point.
Part 2: the barbell lower bound [3a(a-1)l + l^2 + 2]/6 <= tau_2 and the ratios
        K/n^3, tau_2/n^3 for balanced barbells up to n = 600.
Part 3: every inequality of Section 3 on random connected graphs, for several
        thresholds D: k <= s/D, sum of diameters <= 5s/D, B <= 2nD, each tree cut
        <= s^2/8 + B, the contraction bound, and K <= n^3/54 + (37/8) n^(5/2).
Python 3 with numpy.
"""
from fractions import Fraction as Fr
from collections import deque
import random
import numpy as np

out = []
P = out.append


def barbell(a, l):
    """Two copies of K_a joined at vertices 0 and a by a path of l edges."""
    E = [(i, j) for i in range(a) for j in range(i + 1, a)]
    E += [(a + i, a + j) for i in range(a) for j in range(i + 1, a)]
    path = [0] + list(range(2 * a, 2 * a + l - 1)) + [a]
    E += list(zip(path, path[1:]))
    return 2 * a + l - 1, E


def adj(n, E):
    A = [set() for _ in range(n)]
    for u, v in E:
        A[u].add(v)
        A[v].add(u)
    return A


def kemeny_exact(n, E):
    """tr(Z) - 1 with Z = (I - P + 1 pi^T)^{-1}, in exact arithmetic."""
    A = adj(n, E)
    d = [len(x) for x in A]
    W = sum(d)
    M = [[(Fr(1) if i == j else Fr(0)) - (Fr(1, d[i]) if j in A[i] else 0) + Fr(d[j], W)
          for j in range(n)] + [Fr(1) if i == j else Fr(0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return sum(M[i][n + i] for i in range(n)) - 1


def spectra(n, E):
    """(K from the eigenvalues, tau_2, K from the resistance formula)."""
    L = np.zeros((n, n))
    for u, v in E:
        L[u, v] -= 1
        L[v, u] -= 1
        L[u, u] += 1
        L[v, v] += 1
    d = np.diag(L).copy()
    s = 1 / np.sqrt(d)
    mu = np.sort(np.linalg.eigvalsh(s[:, None] * L * s[None, :]))
    Lp = np.linalg.pinv(L)
    g = np.diag(Lp)
    R = g[:, None] + g[None, :] - 2 * Lp
    Kr = float(d @ R @ d) / (2 * d.sum())
    return float(np.sum(1 / mu[1:])), float(1 / mu[1]), Kr, R, d


# ---------------- Part 1 ----------------
P("Part 1 (exact Kemeny constants of barbells B(a,l), n = 2a + l - 1)")
for a, l in ((2, 1), (3, 2), (3, 3), (4, 3), (5, 5), (6, 7)):
    n, E = barbell(a, l)
    Ke = kemeny_exact(n, E)
    Ks, t2, Kr, _, _ = spectra(n, E)
    P("  a=%d l=%d n=%2d  K = %-14s = %.10f  spectral %.10f  resistance %.10f" % (a, l, n, Ke, float(Ke), Ks, Kr))

# ---------------- Part 2 ----------------
P("")
P("Part 2 (balanced barbells a = floor(n/3), l = n - 2a + 1; lower bound L_n = [3a(a-1)l + l^2 + 2]/6)")
for n in (6, 9, 30, 90, 300, 600):
    a = n // 3
    l = n - 2 * a + 1
    nn, E = barbell(a, l)
    assert nn == n
    K, t2, Kr, _, _ = spectra(n, E)
    Ln = Fr(3 * a * (a - 1) * l + l * l + 2, 6)
    low = Fr((n - 2) * (n - 5) * (n + 3), 54)
    P("  n=%3d a=%3d l=%3d  L_n/n^3=%.6f  tau_2/n^3=%.6f  K/n^3=%.6f  L_n<=tau_2: %s  (n-2)(n-5)(n+3)/54<=L_n: %s"
      % (n, a, l, float(Ln) / n ** 3, t2 / n ** 3, K / n ** 3, float(Ln) <= t2 * (1 + 1e-12), low <= Ln))
P("  1/54 = %.6f" % (1 / 54))

# ---------------- Part 3 ----------------
P("")
P("Part 3 (the inequalities of Section 3 on random connected graphs)")


def bfs(A, src, allowed):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in A[u]:
            if v in allowed and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def connected(n, E):
    return len(bfs(adj(n, E), 0, set(range(n)))) == n


def rand_graph(rng):
    kind = rng.choice(["gnp", "cliques+paths", "tree", "lollipop+noise"])
    n = rng.randint(8, 46)
    if kind == "gnp":
        p = rng.choice([0.08, 0.15, 0.3, 0.6])
        E = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    elif kind == "tree":
        E = [(i, rng.randrange(i)) for i in range(1, n)]
    elif kind == "cliques+paths":
        E, v, prev = [], 0, None
        while v < n:
            m = min(rng.choice([1, 1, 2, 5, 9, 14]), n - v)
            blk = list(range(v, v + m))
            E += [(x, y) for i, x in enumerate(blk) for y in blk[i + 1:]]
            if prev is not None:
                E.append((prev, rng.choice(blk)))
            prev = rng.choice(blk)
            v += m
    else:
        m = rng.randint(3, n - 1)
        E = [(i, j) for i in range(m) for j in range(i + 1, m)] + [(i - 1, i) for i in range(m, n)]
        E += [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 3))]
    E = list({(min(u, v), max(u, v)) for u, v in E if u != v})
    return kind, n, E


rng = random.Random(20261106)
counts, worst, tested = {}, 0.0, 0
fails = []
while tested < 400:
    kind, n, E = rand_graph(rng)
    if not connected(n, E):
        continue
    tested += 1
    counts[kind] = counts.get(kind, 0) + 1
    A = adj(n, E)
    K, t2, Kr, R, d = spectra(n, E)
    W = d.sum()
    ok = abs(K - Kr) < 1e-7 * max(1, K) and t2 <= K * (1 + 1e-12)
    bound = n ** 3 / 54 + 37 / 8 * n ** 2.5
    ok &= K <= bound
    worst = max(worst, K / bound)
    for D in (2, 3, np.sqrt(n), n / 4, n / 2):
        H = [v for v in range(n) if d[v] >= D]
        S = set(H) | {u for h in H for u in A[h]}
        comps, seen = [], set()
        for v in sorted(S):
            if v not in seen:
                c = set(bfs(A, v, S))
                seen |= c
                comps.append(c)
        s = len(S)
        k = len(comps)
        diam = [max(max(bfs(A, u, c).values()) for u in c) for c in comps]
        delta = sum(diam)
        # quotient graph: components contracted, outside vertices kept
        part = {}
        for i, c in enumerate(comps):
            for u in c:
                part[u] = i
        nxt = k
        for v in range(n):
            if v not in part:
                part[v] = nxt
                nxt += 1
        Q = adj(nxt, [(part[u], part[v]) for u, v in E if part[u] != part[v]])
        tree = {}  # BFS spanning tree of the quotient: child -> parent
        dq = bfs(Q, 0, set(range(nxt)))
        for x in range(1, nxt):
            tree[x] = next(y for y in Q[x] if dq[y] == dq[x] - 1)
        w = np.zeros(nxt)
        a_ = np.zeros(nxt)
        for v in range(n):
            w[part[v]] += d[v]
        for u, v in E:
            if part[u] == part[v] and part[u] < k:
                a_[part[u]] += 2
        b = w - a_
        Bt = b.sum()
        # weights below each tree edge
        order = sorted(range(1, nxt), key=lambda x: -dq[x])
        below_w, below_a, below_x = w.copy(), a_.copy(), np.zeros(nxt)
        for i, c in enumerate(comps):
            below_x[i] = len(c)
        for x in order:
            p = tree[x]
            below_w[p] += below_w[x]
            below_a[p] += below_a[x]
            below_x[p] += below_x[x]
        cut_sum = 0.0
        for x in range(1, nxt):
            WL, WR = below_w[x], W - below_w[x]
            cut = WL * WR / W
            cut_sum += cut
            ok &= cut <= s * s / 8 + Bt + 1e-9
        # tree distances between quotient vertices
        TA = adj(nxt, list(tree.items()))
        dT = [bfs(TA, x, set(range(nxt))) for x in range(nxt)]
        lhs = 0.0
        for u in range(n):
            for v in range(u + 1, n):
                ok &= R[u, v] <= dT[part[u]][part[v]] + delta + 1e-9  # via R <= dist <= dist_T + delta
        contraction = cut_sum + W * delta / 2
        ok &= K <= contraction + 1e-9
        ok &= (k <= s / D + 1e-12) and (delta <= 5 * s / D + 1e-12) and (Bt <= 2 * n * D + 1e-9)
        ok &= K <= (n - s) * s * s / 8 + 21 / 8 * n ** 3 / D + 2 * n * n * D + 1e-9
    if not ok:
        fails.append((kind, n, len(E)))
P("  graphs tested: %d %s" % (tested, dict(sorted(counts.items()))))
P("  thresholds per graph: D = 2, 3, sqrt(n), n/4, n/2")
P("  all inequalities hold: %s  (failures: %s)" % (not fails, fails[:5]))
P("  largest K / (n^3/54 + (37/8) n^(5/2)) seen: %.4f" % worst)

txt = "\n".join(out) + "\n"
print(txt)
open(__file__.replace(".py", ".out"), "w", encoding="utf-8", newline="\n").write(txt)
