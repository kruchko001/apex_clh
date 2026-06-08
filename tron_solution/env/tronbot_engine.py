import time
import numpy as np

K1, K2 = 55, 194
DEPTH_INITIAL, DEPTH_MAX = 1, 100
DRAW_PENALTY = 0
INF = 10**9

TB_DX = (0, 0, 1, -1)
TB_DY = (-1, 1, 0, 0)
INTERNAL_TO_ACTION = {0: 0, 1: 2, 2: 1, 3: 3}

_POTENTIAL_ARTICULATION = (
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)


def _color(row, col):
    return (row ^ col) & 1


def _num_fillable(red, black, startcolor):
    if startcolor:
        return 2 * min(red - 1, black) + (1 if black >= red else 0)
    return 2 * min(red, black - 1) + (1 if red >= black else 0)


class Components:
    def __init__(self, M):
        self.M = M
        h, w = M.shape
        self.c = np.zeros((h, w), dtype=np.int32)
        self.cedges = []
        self.red = []
        self.black = []
        self.recalc()

    def recalc(self):
        M = self.M
        h, w = M.shape
        equiv = [0]
        self.cedges = []
        self.red = []
        self.black = []
        nextclass = 1
        c = self.c
        c.fill(0)
        mapbottom = w * (h - 1) - 1
        idx = w + 1
        while idx < mapbottom:
            if M.flat[idx]:
                idx += 1
                continue
            cup = equiv[c.flat[idx - w]] if idx >= w else 0
            cleft = equiv[c.flat[idx - 1]] if idx % w else 0
            if cup == 0 and cleft == 0:
                equiv.append(nextclass)
                c.flat[idx] = nextclass
                nextclass += 1
            elif cup == cleft:
                c.flat[idx] = cup
            else:
                if cleft == 0 or (cup != 0 and cup < cleft):
                    c.flat[idx] = cup
                    if cleft != 0:
                        for k in range(len(equiv)):
                            if equiv[k] == cleft:
                                equiv[k] = cup
                else:
                    c.flat[idx] = cleft
                    if cup != 0:
                        for k in range(len(equiv)):
                            if equiv[k] == cup:
                                equiv[k] = cleft
            idx += 1
            if idx % w == w - 1:
                idx += 2
        self.cedges = [0] * nextclass
        self.red = [0] * nextclass
        self.black = [0] * nextclass
        for j in range(1, h - 1):
            for i in range(1, w - 1):
                e = equiv[c[j, i]]
                c[j, i] = e
                self.cedges[e] += self._degree_idx(j * w + i)
                if _color(j, i):
                    self.red[e] += 1
                else:
                    self.black[e] += 1

    def _degree_idx(self, idx):
        M = self.M
        w = M.shape[1]
        return 4 - int(M.flat[idx - 1]) - int(M.flat[idx + 1]) - int(M.flat[idx - w]) - int(M.flat[idx + w])

    def _degree(self, pos):
        r, c = pos
        w = self.M.shape[1]
        return self._degree_idx(r * w + c)

    def _neighbors(self, pos):
        r, c = pos
        M = self.M
        h, w = M.shape
        bits = 0
        for dr, dc, bit in ((-1, -1, 0), (-1, 0, 1), (-1, 1, 2), (0, 1, 3),
                            (1, 1, 4), (1, 0, 5), (1, -1, 6), (0, -1, 7)):
            nr, nc = r + dr, c + dc
            blocked = nr < 0 or nr >= h or nc < 0 or nc >= w or M[nr, nc]
            if blocked:
                bits |= 1 << bit
        return bits

    def _potential_articulation(self, pos):
        return _POTENTIAL_ARTICULATION[self._neighbors(pos)]

    def remove(self, pos):
        comp = int(self.c[pos])
        self.c[pos] = 0
        if self._potential_articulation(pos):
            self.recalc()
        else:
            self.cedges[comp] -= 2 * self._degree(pos)
            if _color(*pos):
                self.red[comp] -= 1
            else:
                self.black[comp] -= 1

    def add(self, pos):
        for m in range(4):
            nr, nc = pos[0] + TB_DY[m], pos[1] + TB_DX[m]
            if nr < 0 or nr >= self.M.shape[0] or nc < 0 or nc >= self.M.shape[1]:
                continue
            if self.M[nr, nc]:
                continue
            comp = self.c[pos]
            ncomp = self.c[nr, nc]
            if comp != 0 and comp != ncomp:
                self.recalc()
                return
            self.c[pos] = ncomp
        comp = self.c[pos]
        self.cedges[comp] += 2 * self._degree(pos)
        if _color(*pos):
            self.red[comp] += 1
        else:
            self.black[comp] += 1

    def component(self, pos):
        return int(self.c[pos])

    def fillablearea(self, pos):
        comp = self.component(pos)
        return _num_fillable(self.red[comp], self.black[comp], _color(*pos))

    def connectedvalue(self, pos):
        return self.cedges[self.component(pos)]


class TronBotEngine:
    def __init__(self, move_timeout=0.05, first_move_timeout=0.15):
        self.move_timeout = move_timeout
        self.first_move_timeout = first_move_timeout
        self._killer = [0] * (DEPTH_MAX * 2 + 2)
        self._first_move = True
        self._timer = 0.0
        self._timeout = 0.0
        self._timed_out = False
        self._maxitr = 0
        self._evaluations = 0

    def reset(self):
        self._killer = [0] * (DEPTH_MAX * 2 + 2)
        self._first_move = True

    def _reset_timer(self, seconds):
        self._timer = time.perf_counter()
        self._timeout = seconds
        self._timed_out = False

    def _poll_timer(self):
        if self._timeout > 0 and (time.perf_counter() - self._timer) > self._timeout:
            self._timed_out = True

    def _wall_grid(self, grid):
        M = np.zeros(grid.shape, dtype=np.int8)
        M[grid != 0] = 1
        return M

    def choose_move(self, grid, bot_pos, opp_pos):
        M = self._wall_grid(grid)
        h, w = M.shape
        for i in range(w):
            M[0, i] = M[h - 1, i] = 1
        for j in range(h):
            M[j, 0] = M[j, w - 1] = 1
        self._reset_timer(self.first_move_timeout if self._first_move else self.move_timeout)
        self._first_move = False
        self._evaluations = 0
        state = {"p": [tuple(bot_pos), tuple(opp_pos)], "m": [0, 0]}
        M[state["p"][0]] = 1
        M[state["p"][1]] = 1
        cp = Components(M)
        if cp.component(state["p"][0]) == cp.component(state["p"][1]):
            internal = self._next_move_alphabeta(M, state)
        else:
            internal = self._next_move_spacefill(M, state, cp)
        return INTERNAL_TO_ACTION.get(internal, 1)

    def _next(self, pos, move):
        return (pos[0] + TB_DY[move], pos[1] + TB_DX[move])

    def _degree(self, M, pos):
        r, c = pos
        h, w = M.shape
        deg = 4
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or M[nr, nc]:
                deg -= 1
        return deg

    def _dijkstra(self, M, start):
        h, w = M.shape
        dist = np.full((h, w), INF, dtype=np.int32)
        q = [[], []]
        active = 0
        dist[start] = 0
        q[0].append(start)
        radius = 0
        while q[active]:
            while q[active]:
                u = q[active].pop()
                assert dist[u] == radius
                for m in range(4):
                    v = self._next(u, m)
                    if v[0] < 0 or v[0] >= h or v[1] < 0 or v[1] >= w or M[v]:
                        continue
                    if dist[v] == INF:
                        dist[v] = radius + 1
                        q[active ^ 1].append(v)
            active ^= 1
            radius += 1
        return dist

    def _floodfill(self, M, cp, pos, fixup=True):
        bestv, best = 0, pos
        for m in range(4):
            p = self._next(pos, m)
            if p[0] < 0 or p[0] >= M.shape[0] or p[1] < 0 or p[1] >= M.shape[1] or M[p]:
                continue
            cp2 = Components(M)
            v = cp2.connectedvalue(p) + cp2.fillablearea(p) - 2 * cp2._degree(p)
            v -= 4 * cp2._potential_articulation(p)
            if v > bestv:
                bestv, best = v, p
        if bestv == 0:
            return 0
        M[best] = 1
        cp.remove(best)
        a = 1 + self._floodfill(M, cp, best)
        M[best] = 0
        if fixup:
            cp.add(best)
        return a

    def _spacefill(self, M, cp, pos, itr):
        if self._degree(M, pos) == 0:
            return 0, 1
        if self._timed_out:
            return 0, 1
        spacesleft = cp.fillablearea(pos)
        if itr == 0:
            return self._floodfill(M, cp, pos), 1
        bestv, bestm = 0, 1
        for m in range(4):
            if self._timed_out:
                break
            r = self._next(pos, m)
            if r[0] < 0 or r[0] >= M.shape[0] or r[1] < 0 or r[1] >= M.shape[1] or M[r]:
                continue
            M[r] = 1
            cp.remove(r)
            v, _ = self._spacefill(M, cp, r, itr - 1)
            v += 1
            M[r] = 0
            cp.add(r)
            if v > bestv:
                bestv, bestm = v, m
            if v == spacesleft:
                break
        return bestv, bestm

    def _next_move_spacefill(self, M, state, cp):
        area = cp.fillablearea(state["p"][0])
        bestv, bestm = 0, 1
        for itr in range(DEPTH_INITIAL, DEPTH_MAX):
            self._poll_timer()
            if self._timed_out:
                break
            self._maxitr = itr
            cp2 = Components(M)
            v, m = self._spacefill(M, cp2, state["p"][0], itr)
            if v > bestv:
                bestv, bestm = v, m
            if v <= itr:
                break
            if v >= area:
                break
        return bestm

    def _reset_articulations(self, h, w):
        self._num = np.zeros((h, w), dtype=np.int32)
        self._low = np.zeros((h, w), dtype=np.int32)
        self._articd = np.zeros((h, w), dtype=np.int8)
        self._art_counter = 0

    def _calc_articulations(self, M, dp0, dp1, v, parent=-1):
        self._art_counter += 1
        nodenum = self._art_counter
        self._low[v] = self._num[v] = nodenum
        children = count = 0
        for m in range(4):
            w = self._next(v, m)
            if w[0] < 0 or w[0] >= M.shape[0] or w[1] < 0 or w[1] >= M.shape[1] or M[w]:
                continue
            if dp0 is not None and dp0[w] >= dp1[w]:
                continue
            if not self._num[w]:
                children += 1
                count += self._calc_articulations(M, dp0, dp1, w, nodenum)
                if self._low[w] >= nodenum and parent != -1:
                    self._articd[v] = 1
                    count += 1
                if self._low[w] < self._low[v]:
                    self._low[v] = self._low[w]
            elif self._num[w] < nodenum:
                if self._num[w] < self._low[v]:
                    self._low[v] = self._num[w]
        if parent == -1 and children > 1:
            count += 1
            self._articd[v] = 1
        return count

    def _explore_space(self, M, dp0, dp1, exits, v):
        c = [0, 0, 0, 0]
        if self._num[v] == 0:
            return c
        if _color(*v):
            c[0] += 1
        else:
            c[1] += 1
        self._num[v] = 0
        if self._articd[v]:
            for m in range(4):
                w = self._next(v, m)
                if w[0] < 0 or w[0] >= M.shape[0] or w[1] < 0 or w[1] >= M.shape[1] or M[w]:
                    continue
                c[2] += 1
                if dp0 is not None and dp0[w] >= dp1[w]:
                    c[3] = 1
                    continue
                if not self._num[w]:
                    continue
                exits.append(w)
        else:
            for m in range(4):
                w = self._next(v, m)
                if w[0] < 0 or w[0] >= M.shape[0] or w[1] < 0 or w[1] >= M.shape[1] or M[w]:
                    continue
                c[2] += 1
                if dp0 is not None and dp0[w] >= dp1[w]:
                    c[3] = 1
                    continue
                if not self._num[w]:
                    continue
                if self._articd[w]:
                    exits.append(w)
                else:
                    child = self._explore_space(M, dp0, dp1, exits, w)
                    for i in range(4):
                        c[i] += child[i]
        return c

    def _max_articulated_space(self, M, dp0, dp1, v):
        exits = []
        space = self._explore_space(M, dp0, dp1, exits, v)
        maxspace = list(space)
        maxsteps = 0
        entrancecolor = _color(*v)
        localsteps = [
            _num_fillable(space[0], space[1] + 1, entrancecolor),
            _num_fillable(space[0] + 1, space[1], entrancecolor),
        ]
        for ex in exits:
            exitcolor = _color(*ex)
            child = self._max_articulated_space(M, dp0, dp1, ex)
            steps = _num_fillable(child[0], child[1], exitcolor)
            if not child[3]:
                steps += localsteps[exitcolor]
            else:
                steps += dp0[ex] - 1
            if steps > maxsteps:
                maxsteps = steps
                if not child[3]:
                    maxspace = [space[i] + child[i] for i in range(4)]
                else:
                    maxspace = child
        return maxspace

    def _evaluate_territory(self, M, state, cp):
        dp0 = self._dijkstra(M, state["p"][0])
        dp1 = self._dijkstra(M, state["p"][1])
        h, w = M.shape
        self._reset_articulations(h, w)
        p0, p1 = state["p"][0], state["p"][1]
        M[p0] = M[p1] = 0
        self._calc_articulations(M, dp0, dp1, p0)
        self._calc_articulations(M, dp1, dp0, p1)
        c0 = self._max_articulated_space(M, dp0, dp1, p0)
        c1 = self._max_articulated_space(M, dp1, dp0, p1)
        nc0 = K1 * (c0[3] + _num_fillable(c0[0], c0[1], _color(*p0))) + K2 * c0[2]
        nc1 = K1 * (c1[3] + _num_fillable(c1[0], c1[1], _color(*p1))) + K2 * c1[2]
        M[p0] = M[p1] = 1
        return nc0 - nc1

    def _evaluate_board(self, M, state, player):
        self._evaluations += 1
        p0, p1 = state["p"][0], state["p"][1]
        if p0 == p1:
            return 0
        M[p0] = M[p1] = 0
        cp = Components(M)
        M[p0] = M[p1] = 1
        if cp.component(p0) == cp.component(p1):
            v = self._evaluate_territory(M, state, cp)
            return v
        h, w = M.shape
        self._reset_articulations(h, w)
        M[p0] = M[p1] = 0
        self._calc_articulations(M, None, None, p0)
        self._calc_articulations(M, None, None, p1)
        c0 = self._max_articulated_space(M, None, None, p0)
        c1 = self._max_articulated_space(M, None, None, p1)
        ff0 = _num_fillable(c0[0], c0[1], _color(*p0))
        ff1 = _num_fillable(c1[0], c1[1], _color(*p1))
        v = 10000 * (ff0 - ff1)
        if v != 0 and abs(v) <= 30000:
            cp2 = Components(M)
            ff0, _ = self._spacefill(M, cp2, p0, 3)
            cp3 = Components(M)
            ff1, _ = self._spacefill(M, cp3, p1, 3)
            v = 10000 * (ff0 - ff1)
        if player == 1:
            v = -v
        M[p0] = M[p1] = 1
        return v

    def _alphabeta(self, M, state, player, a, b, itr, moves):
        moves[0] = 1
        if state["p"][0] == state["p"][1]:
            return DRAW_PENALTY
        dp0 = self._degree(M, state["p"][player])
        dp1 = self._degree(M, state["p"][player ^ 1])
        if dp0 == 0:
            return DRAW_PENALTY if dp1 == 0 else -INF
        if dp1 == 0:
            for m in range(4):
                nxt = self._next(state["p"][player], m)
                if nxt[0] >= 0 and nxt[0] < M.shape[0] and nxt[1] >= 0 and nxt[1] < M.shape[1] and not M[nxt]:
                    moves[0] = m
                    return INF
        self._poll_timer()
        if self._timed_out:
            return a
        if itr == 0:
            return self._evaluate_board(M, state, player)
        kill = self._killer[self._maxitr - itr]
        bestmoves = [0] * len(moves)
        for _m in range(-1, 4):
            if self._timed_out:
                break
            if _m == kill:
                continue
            m = kill if _m == -1 else _m
            nxt = self._next(state["p"][player], m)
            if nxt[0] < 0 or nxt[0] >= M.shape[0] or nxt[1] < 0 or nxt[1] >= M.shape[1] or M[nxt]:
                continue
            r = {"p": list(state["p"]), "m": list(state["m"])}
            r["m"][player] = m
            if player == 1:
                r["p"][0] = self._next(state["p"][0], r["m"][0])
                r["p"][1] = self._next(state["p"][1], r["m"][1])
                M[r["p"][0]] = M[r["p"][1]] = 1
            child_moves = [0] * (len(moves) - 1)
            a_ = -self._alphabeta(M, r, player ^ 1, -b, -a, itr - 1, child_moves)
            if player == 1:
                M[r["p"][0]] = M[r["p"][1]] = 0
            if self._timed_out:
                return -INF
            if a_ > a:
                a = a_
                bestmoves[0] = m
                self._killer[self._maxitr - itr] = m
                bestmoves[1:] = child_moves
            if a >= b:
                break
        moves[:] = bestmoves[: len(moves)]
        return a

    def _next_move_alphabeta(self, M, state):
        lastm = 1
        moves = [0] * (DEPTH_MAX * 2 + 2)
        for itr in range(DEPTH_INITIAL, DEPTH_MAX):
            self._poll_timer()
            if self._timed_out:
                break
            self._maxitr = itr * 2
            v = self._alphabeta(M, state, 0, -INF, INF, itr * 2, moves)
            if v == INF:
                return moves[0]
            if v == -INF:
                break
            lastm = moves[0]
            self._killer[: itr * 2] = moves[: itr * 2]
        self._killer[:-2] = self._killer[2:]
        return lastm
