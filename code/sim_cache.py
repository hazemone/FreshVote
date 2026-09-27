"""Experiment 2: neighbour-cache decisions with periodic beacons and a mid-run quality swap.

Experiment 1 (``sim.py``) delivers a continuous stream of per-tick broadcasts that are
used on arrival, so every candidate message is roughly equally old. This module keeps
the same terrain, obstacle, range/line-of-sight, delay, jitter, loss and quality models,
but changes three things so that message age can differ materially between candidates:

1. Each robot beacons its opinion every ``beacon_period`` ticks (random phase) instead
   of every tick.
2. Each receiver stores the most recent message from every sender in a neighbour cache
   and samples from the cache at every decision; entries are dropped only after a common
   storage retention limit (identical for all policies).
3. The two option qualities swap at ``swap_tick``. Robots re-observe locally, so cached
   messages sent before the swap advertise obsolete opinions and qualities. A small
   spontaneous-switching probability keeps the swarm able to leave a stale consensus.

Motion is vectorised and updated synchronously (all robots move, then headings are
updated), which differs from the sequential loop of Experiment 1 only in update order.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import math
import numpy as np

from config import TerrainConfig, AgentConfig, OptionConfig
from terrain import Terrain
from agent import spawn_agents


@dataclass
class CacheConfig:
    # swarm / world
    num_agents: int = 100
    steps: int = 3000
    dt: float = 0.1
    speed: float = 0.5
    mission_heading: float = math.pi / 2
    heading_noise: float = 0.05
    repulsion_radius: float = 15.0
    repulsion_strength: float = 0.4
    obstacle_radius: float = 3.0
    obstacle_strength: float = 0.6
    ghost_window: int = 50
    # channel
    comm_range: float = 25.0
    distance_delay_scale: float = 10.0
    distance_decay_scale: float = 25.0
    base_delay: int = 4
    jitter: int = 2
    loss: float = 0.0
    beacon_period: int = 10
    # decision
    update_probability: float = 1.0
    switch_probability: float = 0.001
    retention: int = 600
    q_zero: float = 0.60
    q_one: float = 0.75
    noise: float = 0.05
    swap_tick: int = 1500
    # policy
    policy: str = "baseline"          # baseline | hard_ttl | fixed_decay | adaptive_freshvote
    ttl: float = 20.0
    lam: float = 0.05
    kappa: float = 1.5
    min_ttl: float = 2.0
    cold_start: int = 3
    # outcome
    threshold: float = 0.90
    persistence: int = 100
    # seeds
    seed: int = 1


def _los_batch(grid: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Vectorised line-of-sight by sampling the segment every ``step`` metres (cell size 1 m)."""
    if len(p1) == 0:
        return np.zeros(0, dtype=bool)
    depth, width = grid.shape
    d = np.linalg.norm(p2 - p1, axis=1)
    n = int(np.ceil(d.max() / step)) + 1
    s = np.linspace(0.0, 1.0, n)
    xs = p1[:, 0:1] + (p2[:, 0:1] - p1[:, 0:1]) * s
    ys = p1[:, 1:2] + (p2[:, 1:2] - p1[:, 1:2]) * s
    cols = np.clip(xs.astype(int), 0, width - 1)
    rows = np.clip(ys.astype(int), 0, depth - 1)
    return ~grid[rows, cols].any(axis=1)


def _row_quantiles(ages: np.ndarray, valid: np.ndarray, qs=(0.25, 0.5, 0.75)):
    """Linear-interpolated quantiles (numpy default) of the valid entries in each row."""
    big = np.where(valid, ages, np.inf)
    srt = np.sort(big, axis=1)
    n = valid.sum(axis=1)
    out = []
    rows = np.arange(len(n))
    for q in qs:
        pos = np.maximum(n - 1, 0) * q
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        frac = pos - lo
        vlo = srt[rows, lo]
        vhi = srt[rows, hi]
        with np.errstate(invalid="ignore"):
            val = np.where(n > 0, vlo + (vhi - vlo) * frac, np.nan)
        out.append(val)
    return out, n


def policy_weights(cfg: CacheConfig, ages: np.ndarray, q: np.ndarray, valid: np.ndarray):
    """Return weights (rows = deciding robots, cols = cached senders)."""
    base = np.where(valid, np.maximum(q, 0.0), 0.0)
    if cfg.policy == "baseline":
        return base
    if cfg.policy == "hard_ttl":
        return base * (ages <= cfg.ttl)
    if cfg.policy == "fixed_decay":
        return base * np.exp(-cfg.lam * ages)
    if cfg.policy == "adaptive_freshvote":
        (q25, med, q75), n = _row_quantiles(ages, valid)
        ttl = np.maximum(cfg.min_ttl, med + cfg.kappa * (q75 - q25))
        lam = np.log(2.0) / np.maximum(med, 1.0)
        w = base * np.exp(-lam[:, None] * ages) * (ages <= ttl[:, None])
        cold = n < cfg.cold_start
        w[cold] = base[cold]
        return w
    raise ValueError(cfg.policy)


def run_cache(cfg: CacheConfig, record_every: int = 10, keep_series: bool = False) -> Dict:
    seed = cfg.seed
    terrain = Terrain(TerrainConfig(seed=30_000 + seed))
    grid = terrain.grid
    depth, width = grid.shape
    ss = np.random.SeedSequence(50_000 + seed)
    spawn_rng, comms_rng, cons_rng, dyn_rng = [np.random.default_rng(s) for s in ss.spawn(4)]
    agents = spawn_agents(AgentConfig(num_agents=cfg.num_agents), TerrainConfig(seed=30_000 + seed), terrain,
                          spawn_rng, mission_heading=cfg.mission_heading,
                          option_cfg=OptionConfig(enabled=True, quality_zero=cfg.q_zero, quality_one=cfg.q_one,
                                                  measurement_noise_halfwidth=cfg.noise))
    N = cfg.num_agents
    pos = np.array([a.position[:2] for a in agents], dtype=np.float64)
    heading = np.array([a.heading for a in agents], dtype=np.float64)
    opinion = np.array([a.opinion for a in agents], dtype=np.int64)
    # fixed local measurement errors so that re-observation after the swap is consistent
    eps = np.stack([np.array([a.quality_obs_zero for a in agents]) - cfg.q_zero,
                    np.array([a.quality_obs_one for a in agents]) - cfg.q_one], axis=1)
    alive = np.ones(N, dtype=bool)
    mobile = np.ones(N, dtype=bool)
    stuck_since = np.full(N, -1, dtype=np.int64)
    phase = comms_rng.integers(0, cfg.beacon_period, size=N)

    cache_send = np.full((N, N), -1, dtype=np.int64)
    cache_op = np.zeros((N, N), dtype=np.int64)
    cache_q = np.zeros((N, N), dtype=np.float64)
    pending: Dict[int, list] = {}

    Q = np.array([cfg.q_zero, cfg.q_one])
    best = int(cfg.q_one > cfg.q_zero)
    eye = np.eye(N, dtype=bool)

    series_t, series_c = [], []
    correct_hist = np.zeros(cfg.steps)
    stats = dict(candidates=0, accepted=0, decisions=0, holds=0, switches=0,
                 age_sum=0.0, stale_candidates=0, cache_size_sum=0.0, delivered=0, sent=0)
    fresh_bound = cfg.beacon_period + cfg.base_delay + cfg.jitter + math.ceil(cfg.comm_range / cfg.distance_delay_scale)

    ox, oy = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    ox = ox.ravel(); oy = oy.ravel()

    for t in range(cfg.steps):
        if t == cfg.swap_tick:
            Q = Q[::-1].copy()
            best = 1 - best
        # ghosts expire after the recognition window
        expire = alive & ~mobile & (stuck_since >= 0) & (t - stuck_since >= cfg.ghost_window)
        alive &= ~expire

        # ---- beacons -------------------------------------------------------
        senders = np.flatnonzero(alive & ((t - phase) % cfg.beacon_period == 0))
        if len(senders):
            diff = pos[senders][:, None, :] - pos[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            inr = (dist <= cfg.comm_range) & alive[None, :]
            inr[np.arange(len(senders)), senders] = False
            si, rj = np.nonzero(inr)
            if len(si):
                s_ids = senders[si]
                ok = _los_batch(grid, pos[s_ids], pos[rj])
                s_ids, rj, dd = s_ids[ok], rj[ok], dist[si[ok], rj[ok]]
                stats["sent"] += len(rj)
                if cfg.loss > 0:
                    keep = comms_rng.random(len(rj)) >= cfg.loss
                    s_ids, rj, dd = s_ids[keep], rj[keep], dd[keep]
                if len(rj):
                    jit = comms_rng.integers(0, cfg.jitter + 1, size=len(rj)) if cfg.jitter > 0 else 0
                    dtime = t + cfg.base_delay + np.ceil(dd / cfg.distance_delay_scale).astype(int) + jit
                    op = opinion[s_ids]
                    qobs = np.clip(Q[op] + eps[s_ids, op], 0.0, 1.0)
                    ghost = np.where(mobile[s_ids] | (stuck_since[s_ids] < 0), 1.0,
                                     np.maximum(0.0, 1.0 - (t - stuck_since[s_ids]) / cfg.ghost_window))
                    qeff = np.clip(qobs * ghost * np.exp(-dd / cfg.distance_decay_scale), 0.0, 1.0)
                    dtime = np.broadcast_to(dtime, rj.shape)
                    for dt_ in np.unique(dtime):
                        m = dtime == dt_
                        pending.setdefault(int(dt_), []).append((rj[m], s_ids[m], op[m], qeff[m], t))

        # ---- delivery into the neighbour cache ------------------------------
        for r, s, op, qq, st in pending.pop(t, []):
            stats["delivered"] += len(r)
            newer = st > cache_send[r, s]
            r, s, op, qq = r[newer], s[newer], op[newer], qq[newer]
            cache_send[r, s] = st
            cache_op[r, s] = op
            cache_q[r, s] = qq

        # ---- decisions -----------------------------------------------------
        u = alive & mobile & (cons_rng.random(N) < cfg.update_probability)
        sw = u & (cons_rng.random(N) < cfg.switch_probability)
        opinion[sw] = 1 - opinion[sw]
        stats["switches"] += int(sw.sum())
        dec = np.flatnonzero(u & ~sw)
        if len(dec):
            ages = (t - cache_send[dec]).astype(np.float64)
            valid = (cache_send[dec] >= 0) & (ages <= cfg.retention) & ~eye[dec]
            w = policy_weights(cfg, ages, cache_q[dec], valid)
            pos_w = w > 0
            stats["candidates"] += int(valid.sum())
            stats["accepted"] += int(pos_w.sum())
            stats["age_sum"] += float(ages[pos_w].sum())
            stats["stale_candidates"] += int((valid & (ages > fresh_bound)).sum())
            stats["cache_size_sum"] += float(valid.sum(axis=1).mean())
            stats["decisions"] += len(dec)
            tot = w.sum(axis=1)
            has = tot > 0
            stats["holds"] += int((~has).sum())
            if has.any():
                cw = np.cumsum(w[has], axis=1)
                rr = cons_rng.random(int(has.sum())) * tot[has]
                idx = (cw > rr[:, None]).argmax(axis=1)
                rows = dec[has]
                opinion[rows] = cache_op[rows, idx]

        # ---- motion (synchronous) --------------------------------------------
        mv = np.flatnonzero(alive & mobile)
        if len(mv):
            nx = np.clip(pos[mv, 0] + cfg.speed * np.cos(heading[mv]) * cfg.dt, 1, width - 1)
            ny = np.clip(pos[mv, 1] + cfg.speed * np.sin(heading[mv]) * cfg.dt, 1, depth - 1)
            hit = grid[np.clip(ny.astype(int), 0, depth - 1), np.clip(nx.astype(int), 0, width - 1)]
            stuck = mv[hit]
            mobile[stuck] = False
            stuck_since[stuck] = t
            go = mv[~hit]
            pos[go, 0] = nx[~hit]
            pos[go, 1] = ny[~hit]
            if len(go):
                err = (cfg.mission_heading - heading[go] + np.pi) % (2 * np.pi) - np.pi
                heading[go] += 0.1 * err + dyn_rng.uniform(-cfg.heading_noise, cfg.heading_noise, size=len(go))
                # repulsion from mobile neighbours
                mob = np.flatnonzero(alive & mobile)
                d = pos[go][:, None, :] - pos[mob][None, :, :]
                dist = np.sqrt((d ** 2).sum(-1))
                wgt = np.where((dist > 0) & (dist < cfg.repulsion_radius),
                               (cfg.repulsion_radius - dist) / cfg.repulsion_radius, 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ux = np.where(dist > 0, d[..., 0] / dist, 0.0)
                    uy = np.where(dist > 0, d[..., 1] / dist, 0.0)
                rx = (ux * wgt).sum(1); ry = (uy * wgt).sum(1)
                nz = (rx != 0) | (ry != 0)
                ang = np.arctan2(ry, rx)
                heading[go] += np.where(nz, cfg.repulsion_strength *
                                        ((ang - heading[go] + np.pi) % (2 * np.pi) - np.pi), 0.0)
                # obstacle avoidance from nearby obstacle cells
                row = np.clip(pos[go, 1].astype(int), 0, depth - 1)
                col = np.clip(pos[go, 0].astype(int), 0, width - 1)
                rr_ = row[:, None] + oy[None, :]
                cc_ = col[:, None] + ox[None, :]
                inb = (rr_ >= 0) & (rr_ < depth) & (cc_ >= 0) & (cc_ < width)
                occ = inb & grid[np.clip(rr_, 0, depth - 1), np.clip(cc_, 0, width - 1)]
                dx = pos[go, 0:1] - (cc_ + 0.5)
                dy = pos[go, 1:2] - (rr_ + 0.5)
                dd = np.sqrt(dx ** 2 + dy ** 2)
                ww = np.where(occ & (dd > 0) & (dd <= cfg.obstacle_radius),
                              (cfg.obstacle_radius - dd) / cfg.obstacle_radius, 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ax = np.where(dd > 0, dx / dd, 0.0) * ww
                    ay = np.where(dd > 0, dy / dd, 0.0) * ww
                ax = ax.sum(1); ay = ay.sum(1)
                nz = (ax != 0) | (ay != 0)
                ang = np.arctan2(ay, ax)
                heading[go] += np.where(nz, cfg.obstacle_strength *
                                        ((ang - heading[go] + np.pi) % (2 * np.pi) - np.pi), 0.0)

        # ---- metrics -----------------------------------------------------------
        na = alive.sum()
        cfrac = float((opinion[alive] == best).mean()) if na else 0.0
        correct_hist[t] = cfrac
        if keep_series and t % record_every == 0:
            series_t.append(t); series_c.append(cfrac)

    def first_sustained(start: int, end: int) -> Optional[int]:
        ok = correct_hist[start:end] >= cfg.threshold
        run = 0
        for k, v in enumerate(ok):
            run = run + 1 if v else 0
            if run >= cfg.persistence:
                return start + k - cfg.persistence + 1
        return None

    pre = first_sustained(0, cfg.swap_tick)
    post = first_sustained(cfg.swap_tick, cfg.steps)
    out = dict(asdict(cfg))
    out.update(
        pre_decision_tick=pre,
        recovery_ticks=(post - cfg.swap_tick) if post is not None else None,
        recovered=post is not None,
        pre_mean_correct=float(correct_hist[:cfg.swap_tick].mean()),
        post_mean_correct=float(correct_hist[cfg.swap_tick:].mean()),
        final_correct=float(correct_hist[-1]),
        acceptance=stats["accepted"] / stats["candidates"] if stats["candidates"] else 0.0,
        mean_accepted_age=stats["age_sum"] / stats["accepted"] if stats["accepted"] else 0.0,
        stale_share=stats["stale_candidates"] / stats["candidates"] if stats["candidates"] else 0.0,
        mean_cache_size=stats["cache_size_sum"] / max(1, cfg.steps),
        hold_share=stats["holds"] / stats["decisions"] if stats["decisions"] else 0.0,
        switches=stats["switches"], delivered=stats["delivered"], sent=stats["sent"],
        final_alive=int(alive.sum()), final_mobile=int((alive & mobile).sum()),
    )
    if keep_series:
        out["series_t"] = series_t
        out["series_c"] = series_c
    return out


if __name__ == "__main__":
    import time
    for pol in ("baseline", "adaptive_freshvote"):
        t0 = time.time()
        r = run_cache(CacheConfig(policy=pol, seed=1))
        print(pol, round(time.time() - t0, 1), "s", {k: r[k] for k in (
            "pre_decision_tick", "recovery_ticks", "pre_mean_correct", "post_mean_correct",
            "acceptance", "mean_accepted_age", "stale_share", "mean_cache_size", "hold_share")})
