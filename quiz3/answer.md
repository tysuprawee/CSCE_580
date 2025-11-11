# Quiz 3 Answers

## Q1: Search and Heuristics
- **(a)** An admissible heuristic never overestimates the true minimum cost from the current state to a goal; it is optimistic so the estimated cost is always less than or equal to the optimal remaining path cost.
- **(b)** The heuristic `h = 0` is admissible because it never exceeds the actual remaining cost; it simply reduces a best-first search to uniform-cost or breadth-first search depending on edge costs.
- **(c)** The family of heuristics `{h, h + 1, h + 2, ..., h + k}` is admissible exactly when the base heuristic `h` is admissible *and* every additive offset is non-positive. Adding any positive constant shifts the estimates above those of `h`, so for states whose true cost is equal to `h` the adjusted heuristic would overestimate; conversely, adding zero leaves admissibility intact, while subtracting constants keeps the estimate below or equal to the true cost.
- **(d)** When `h1` is admissible, the composed heuristics `h2 = h1 + 1` and `h3 = h1 + 4` are inadmissible in general because the positive offsets create states for which `hi(s) > h*(s)`. Only `h1` remains admissible; to preserve admissibility, any derived heuristic must not increase the estimate above `h1`'s values.
- **(e)** Taking the minimum of heuristics, `h = min(h1, h2, h3)`, is admissible if at least one of the component heuristics is admissible because the minimum can never exceed that admissible estimate. Taking the maximum, `h = max(h1, h2, h3)`, is admissible only when *all* component heuristics are admissible; if any heuristic overestimates, the maximum can inherit that inadmissibility.

## Q2: Missionaries and Cannibals Search
- **(Q2.1)** In the breadth-first search formulation each state records `(left_missionaries, left_cannibals, boat_position)` with the right-bank counts inferred from the totals. The goal test checks for `(0, 0, "right")`, meaning everyone has safely reached the right bank and the boat ends there. Every action moves one or two passengers; the path-cost function increments by one per crossing so the objective is to minimize the number of boat trips.
- **(Q2.2)** I reimplemented the solver with an explicit BFS frontier in `mc_agent.py`. States store the number of missionaries and cannibals on the left bank plus a flag for the boat location, and the queue explores nodes in order of increasing crossing count. Successor states are generated in a deterministic order so the first solution dequeued matches the canonical sequences shown in the provided notebook. Parent links reconstruct the sequence of moves once the goal `(0, 0, boat_right)` is popped from the queue.
- **(Q2.3)** Running the BFS agent on the six test cases from `MCTester` shows the move sequences and timings below (crossings equal the number of moves).
  - 1M/1C → 1 crossing, moves `[(1, 1)]`, time `0.000016s`
  - 2M/2C → 5 crossings, moves `[(0, 2), (0, 1), (2, 0), (1, 0), (1, 1)]`, time `0.000041s`
  - 3M/3C → 11 crossings, moves `[(0, 2), (0, 1), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]`, time `0.000054s`
  - 4M/3C → 11 crossings, moves `[(0, 2), (0, 1), (2, 0), (1, 0), (1, 1), (0, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]`, time `0.000082s`
  - 5M/3C → 13 crossings, moves `[(2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (2, 0), (1, 0), (1, 1)]`, time `0.000114s`
  - 2M/3C → No solution (empty move list), time `0.000001s`
- **(Q2.4)** The tester outputs show that the breadth-first policy finds the shortest sequences by exhausting all partial plans at a given depth before proceeding. The canonical solution pattern repeatedly ferries two cannibals across, then returns one cannibal, interleaving occasional missionary trips to respect the safety constraint. Those alternating transfers minimize backtracking and keep the frontier aligned with the notebook’s expected move ordering across the tested problem sizes.

## Q3: Cryptanalyst CSP
- **Variables:** `T, W, O, F, U, R` corresponding to each distinct letter in the sum, plus carry variables `c1, c2, c3` for the ones, tens, and hundreds columns.
- **Domains:** Each letter variable ranges over digits `0–9` with an all-different constraint; `T` and `F` cannot be zero because they lead numbers; carries `c1, c2` are in `{0, 1}`, and `c3` is in `{0, 1}` because adding two three-digit numbers yields at most a four-digit result.
- **Constraints:**
  - Column equations: `O + O = R + 10*c1`, `W + W + c1 = U + 10*c2`, `T + T + c2 = O + 10*c3`, and `c3 = F` to satisfy the thousands column.
  - All-different constraint across `T, W, O, F, U, R`.
  - Domain restrictions for leading digits (`T ≠ 0`, `F ≠ 0`).

- **(Q3.b)** Applying arc consistency can shrink domains before search. Pseudo-code sketch: enforce AC-3 over the binary constraints derived from the column equations and the all-different decomposition; propagate reductions iteratively until no domain changes occur.
  - Initialize domains per the constraints above.
  - Place all arcs `(Xi, Xj)` derived from binary constraints (e.g., each pair in the all-different constraint, equation relationships between letters and carries) into a queue.
  - While the queue is non-empty, revise the domain of `Xi` relative to `Xj`; if a domain shrinks, enqueue neighboring arcs of `Xi`.
  - Stop when no domain changes; if any domain becomes empty, the CSP is inconsistent.