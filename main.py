import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

DRIFT_BY_AGE = [
    -0.46, -0.29, -0.19, -0.11, -0.13, -0.086, -0.063, -0.043, 0.028, 0.067,
    0.087, 0.13, 0.20, 0.13, 0.22, 0.15, 0.15, 0.17, 0.18, 0.17, 0.24,
]
DRIFT_20_PLUS = 0.24

SAMPLE_TIMES = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
TIME_WEIGHTS = [math.exp(-t / 24.0) for t in SAMPLE_TIMES]
TW_SUM = sum(TIME_WEIGHTS)
TIME_WEIGHTS = [w / TW_SUM for w in TIME_WEIGHTS]
EMPIRICAL_SAMPLE_TIMES = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
EMPIRICAL_TIME_WEIGHTS = [math.exp(-t / 48.0) for t in EMPIRICAL_SAMPLE_TIMES]
ETW_SUM = sum(EMPIRICAL_TIME_WEIGHTS)
EMPIRICAL_TIME_WEIGHTS = [w / ETW_SUM for w in EMPIRICAL_TIME_WEIGHTS]

LANDING_WEIGHTS = {
    1: 0.21, 2: 0.11, 3: 0.04, 4: 0.02, 5: 0.02,
    6: 0.11, 7: 0.11, 8: 0.06, 9: 0.17, 10: 0.15,
}

NAMED_MASS = 0.90
SURGE_CAP = 0.6 * NAMED_MASS
FLOOR_MIX = 0.12
FLOOR_N = 15
MIN_P = 0.001
ENSEMBLE_DIFF = 0.8
ENSEMBLE_EMP = 0.2
LOGISTIC_LR = 0.5
LOGISTIC_ITERS = 1000
LOGISTIC_L2 = 0.01
LOGISTIC_CLAMP = (-30.0, 30.0)

RANK_BUCKETS = [(1, 3), (4, 10), (11, 20), (21, 30), (31, 50), (51, 100)]
AGE_BUCKETS = [(0, 2), (3, 6), (7, 14), (15, float("inf"))]


def parse_ts(s):
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def drift(age):
    if age <= 0:
        return DRIFT_BY_AGE[0]
    if age >= len(DRIFT_BY_AGE) - 1:
        return DRIFT_BY_AGE[-1]
    lo = int(math.floor(age))
    hi = lo + 1
    return DRIFT_BY_AGE[lo] + (age - lo) * (DRIFT_BY_AGE[hi] - DRIFT_BY_AGE[lo])


def gaussian_pdf(x, mu, sigma):
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def bucket_rank(r):
    for lo, hi in RANK_BUCKETS:
        if lo <= r <= hi:
            return (lo, hi)
    return None


def bucket_age(a):
    for lo, hi in AGE_BUCKETS:
        if lo <= a <= hi:
            return (lo, hi)
    return None


def main():
    history_path = os.path.join(INPUT_DIR, "history.ndjson")
    task_path = os.path.join(INPUT_DIR, "task.json")

    snapshots = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                at = parse_ts(obj.get("at", ""))
                if at is None:
                    continue
                games = []
                for g in obj.get("up_and_coming", []):
                    try:
                        rank = int(g["rank"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    try:
                        uid = int(g["universe_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    playing = g.get("playing")
                    if playing is None:
                        playing_val = None
                    else:
                        try:
                            playing_val = int(playing)
                        except (TypeError, ValueError):
                            playing_val = None
                    games.append({"rank": rank, "universe_id": uid, "playing": playing_val})
                snapshots.append({"at": at, "games": games})

    if not snapshots:
        out = {"probabilities": [], "top10": []}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "forecast.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
        return

    snapshots.sort(key=lambda s: s["at"])
    n_snap = len(snapshots)
    last = snapshots[-1]

    game_first = {}
    game_last_rank = {}
    game_last_playing = {}
    game_rank_series = {}
    game_playing_series = {}

    for idx, snap in enumerate(snapshots):
        for g in snap["games"]:
            uid = g["universe_id"]
            if uid not in game_first:
                game_first[uid] = idx
            game_last_rank[uid] = g["rank"]
            game_last_playing[uid] = g["playing"]
            game_rank_series.setdefault(uid, [])
            game_rank_series[uid].append((idx, g["rank"]))
            game_playing_series.setdefault(uid, [])
            game_playing_series[uid].append((idx, g["playing"]))

    def last_seen_rank(uid):
        rs = game_rank_series.get(uid, [])
        if rs:
            return rs[-1][1]
        return 50

    def chart_age(uid):
        fa = game_first.get(uid, n_snap - 1)
        if fa == 0:
            return 20
        return n_snap - 1 - fa

    def playing_at(uid, idx):
        ps = game_playing_series.get(uid, [])
        for i, p in ps:
            if i == idx:
                return p
        return None

    def median3(vals):
        if not vals:
            return None
        s = sorted(vals)
        return s[len(s) // 2]

    def median_prev3(uid, last_idx):
        vals = []
        for i, p in game_playing_series.get(uid, []):
            if i in (last_idx - 3, last_idx - 2, last_idx - 1) and p is not None:
                vals.append(p)
        if not vals:
            return None
        return median3(vals)

    def spike_at(uid, idx):
        vals = []
        for i, p in game_playing_series.get(uid, []):
            if i in (idx - 3, idx - 2, idx - 1) and p is not None and p > 0:
                vals.append(p)
        if not vals:
            return 0.0
        med = median3(vals)
        if med <= 0:
            return 0.0
        p_now = playing_at(uid, idx)
        if p_now is None:
            return 0.0
        return p_now / med

    def mean_gap_hours():
        if n_snap < 2:
            return 6.0
        total = 0.0
        for i in range(1, n_snap):
            total += (snapshots[i]["at"] - snapshots[i - 1]["at"]).total_seconds() / 3600.0
        return total / (n_snap - 1)

    mean_gap = mean_gap_hours()

    # --- Step 2: candidates ---
    last_rank_set = {g["universe_id"]: g["rank"] for g in last["games"]}
    candidate_set = set()
    for uid, r in last_rank_set.items():
        if r <= 100:
            candidate_set.add(uid)
    for snap in snapshots[-4:]:
        for g in snap["games"]:
            if g["rank"] <= 15:
                candidate_set.add(g["universe_id"])
    candidates = sorted(candidate_set)

    if not candidates:
        out = {"probabilities": [], "top10": []}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "forecast.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
        return

    # --- Step 3: diffusion component ---
    diff_scores = {uid: [0.0] * 11 for uid in candidates}
    expected_rank = {}
    for uid in candidates:
        r0 = last_seen_rank(uid)
        age = chart_age(uid)
        drift_rate = drift(age) * 2.0
        pred_means = []
        score = [0.0] * 11
        for t, tw in zip(SAMPLE_TIMES, TIME_WEIGHTS):
            mu = r0 + drift_rate * 12.0 * (1.0 - math.exp(-t / 12.0))
            mu = max(1.0, min(120.0, mu))
            pred_means.append(mu)
            sigma = math.sqrt(1.5 ** 2 + 0.5 ** 2 * t)
            for k in range(1, 11):
                score[k] += gaussian_pdf(k, mu, sigma) * tw
        expected_rank[uid] = sum(pred_means) / len(pred_means)
        for k in range(1, 11):
            diff_scores[uid][k] = score[k]

    # --- Step 4: surge ---
    surge_mass = {uid: [0.0] * 11 for uid in candidates}
    has_surge = {}
    for uid in candidates:
        p_now = playing_at(uid, n_snap - 1)
        med = median_prev3(uid, n_snap - 1)
        if p_now is None or med is None or med <= 0:
            has_surge[uid] = False
            continue
        spike = p_now / med
        if spike < 3.0:
            has_surge[uid] = False
            continue
        strength = min((spike - 3.0) / (5.0 - 3.0), 1.0)
        has_surge[uid] = True
        for k in range(1, 11):
            surge_mass[uid][k] = 0.7 * strength * LANDING_WEIGHTS[k]

    for k in range(1, 11):
        surge_total = sum(surge_mass[uid][k] for uid in candidates)
        capped = min(surge_total, SURGE_CAP)
        for uid in candidates:
            if has_surge[uid]:
                diff_scores[uid][k] = diff_scores[uid][k] * (1.0 - capped / NAMED_MASS) + surge_mass[uid][k]
            else:
                diff_scores[uid][k] = diff_scores[uid][k] * NAMED_MASS

    # --- Step 5: floor ---
    sorted_by_exp = sorted(candidates, key=lambda uid: expected_rank[uid])
    floor_ids = set(sorted_by_exp[:FLOOR_N])
    place_totals = {}
    for k in range(1, 11):
        place_totals[k] = sum(diff_scores[uid][k] for uid in candidates)
    for uid in candidates:
        for k in range(1, 11):
            if uid in floor_ids:
                diff_scores[uid][k] = 0.88 * diff_scores[uid][k] + 0.12 * (place_totals[k] / FLOOR_N)
            else:
                diff_scores[uid][k] = 0.88 * diff_scores[uid][k]

    for k in range(1, 11):
        for uid in candidates:
            if diff_scores[uid][k] < MIN_P:
                diff_scores[uid][k] = 0.0
        tot = sum(diff_scores[uid][k] for uid in candidates)
        if tot > 0:
            for uid in candidates:
                diff_scores[uid][k] /= tot
        else:
            for uid in candidates:
                diff_scores[uid][k] = 1.0 / len(candidates)

    # --- Step 6: empirical transition ---
    lag_counts = {}
    age_marginals = {}
    for i in range(n_snap):
        snap_i = snapshots[i]
        games_i = {g["universe_id"]: g["rank"] for g in snap_i["games"]}
        for j in range(i + 1, n_snap):
            snap_j = snapshots[j]
            games_j = {g["universe_id"]: g["rank"] for g in snap_j["games"]}
            for uid, r_i in games_i.items():
                r_j = games_j.get(uid, None)
                age_i = i - game_first.get(uid, i)
                if age_i == 0:
                    age_i = 20
                rb = bucket_rank(r_i)
                ab = bucket_age(age_i)
                if rb is None or ab is None:
                    continue
                lag = j - i
                key = (rb, ab, lag)
                if key not in lag_counts:
                    lag_counts[key] = {}
                if r_j is None:
                    rj_key = "off"
                else:
                    rj_key = r_j
                lag_counts[key][rj_key] = lag_counts[key].get(rj_key, 0) + 1

                if (rb, lag) not in age_marginals:
                    age_marginals[(rb, lag)] = {}
                if r_j is None:
                    rj_key = "off"
                else:
                    rj_key = r_j
                age_marginals[(rb, lag)][rj_key] = age_marginals[(rb, lag)].get(rj_key, 0) + 1

    def smoothed_probs(rb, ab, lag):
        cell = lag_counts.get((rb, ab, lag), {})
        marg = age_marginals.get((rb, lag), {})
        cell_total = sum(cell.values())
        marg_total = sum(marg.values())
        probs = {}
        for r in list(range(1, 121)) + ["off"]:
            c = cell.get(r, 0)
            m = marg.get(r, 0)
            s = c + 40 * (m / marg_total) if marg_total > 0 else c
            probs[r] = s / (cell_total + 40) if (cell_total + 40) > 0 else 0.0
        return probs

    emp_scores = {uid: [0.0] * 11 for uid in candidates}
    for uid in candidates:
        r0 = last_seen_rank(uid)
        age = chart_age(uid)
        rb = bucket_rank(r0)
        ab = bucket_age(age)
        if rb is None or ab is None:
            continue
        score = [0.0] * 11
        for t, tw in zip(EMPIRICAL_SAMPLE_TIMES, EMPIRICAL_TIME_WEIGHTS):
            lag = round(t / mean_gap)
            lag = max(1, min(13, lag))
            probs = smoothed_probs(rb, ab, lag)
            for k in range(1, 11):
                score[k] += probs.get(k, 0.0) * tw
        for k in range(1, 11):
            emp_scores[uid][k] = score[k]

    for k in range(1, 11):
        tot = sum(emp_scores[uid][k] for uid in candidates)
        if tot > 0:
            for uid in candidates:
                emp_scores[uid][k] /= tot
        else:
            for uid in candidates:
                emp_scores[uid][k] = 1.0 / len(candidates)

    for k in range(1, 11):
        for uid in candidates:
            if emp_scores[uid][k] * NAMED_MASS < MIN_P:
                emp_scores[uid][k] = 0.0
        tot = sum(emp_scores[uid][k] for uid in candidates)
        if tot > 0:
            for uid in candidates:
                emp_scores[uid][k] /= tot
        else:
            for uid in candidates:
                emp_scores[uid][k] = 1.0 / len(candidates)

    # --- Step 7: ensemble ---
    mixed = {uid: [0.0] * 11 for uid in candidates}
    for uid in candidates:
        for k in range(1, 11):
            mixed[uid][k] = ENSEMBLE_DIFF * diff_scores[uid][k] + ENSEMBLE_EMP * emp_scores[uid][k]

    # --- Step 8: logistic tilt ---
    train_examples = []
    for i in range(n_snap - 1):
        snap_i = snapshots[i]
        games_i = {g["universe_id"]: g["rank"] for g in snap_i["games"]}
        top10_i = sorted(games_i.items(), key=lambda kv: kv[1])[:10]
        top10_ids_i = {uid for uid, _ in top10_i}
        for uid, r_i in games_i.items():
            if r_i > 100 or uid in top10_ids_i:
                continue
            label = 0
            for j in range(i + 1, min(i + 13, n_snap)):
                snap_j = snapshots[j]
                games_j = {g["universe_id"]: g["rank"] for g in snap_j["games"]}
                if uid in games_j and games_j[uid] <= 10:
                    label = 1
                    break
            age_i = i - game_first.get(uid, i)
            p_now = playing_at(uid, i)
            log10_play = math.log10(p_now) if p_now is not None and p_now > 0 else 0.0
            spk = spike_at(uid, i)
            p_prev3 = playing_at(uid, i - 3)
            growth = (p_now / p_prev3) if (p_now is not None and p_prev3 is not None and p_prev3 > 0) else 0.0
            r_prev3 = None
            if i - 3 >= 0:
                r_prev3 = games_i.get(uid) if False else None
                for g in snapshots[i - 3]["games"]:
                    if g["universe_id"] == uid:
                        r_prev3 = g["rank"]
                        break
            mom = (r_prev3 - r_i) if r_prev3 is not None else 0
            train_examples.append([[float(r_i), float(age_i), log10_play, float(spk), float(growth), float(mom)], label])

    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    n_feat = 6
    weights = [0.0] * n_feat
    bias = 0.0

    def standardize(feats, mean, std):
        return [(f - m) / (s + 1e-9) for f, m, s in zip(feats, mean, std)]

    if train_examples:
        n_ex = len(train_examples)
        feat_matrix = [ex[0] for ex in train_examples]
        means = [0.0] * n_feat
        stds = [0.0] * n_feat
        for f in feat_matrix:
            for d in range(n_feat):
                means[d] += f[d]
        means = [m / n_ex for m in means]
        for f in feat_matrix:
            for d in range(n_feat):
                stds[d] += (f[d] - means[d]) ** 2
        stds = [math.sqrt(s / n_ex) for s in stds]

        train_examples_std = [[standardize(ex[0], means, stds), ex[1]] for ex in train_examples]

        n_ex = len(train_examples_std)
        for _ in range(LOGISTIC_ITERS):
            grad_w = [0.0] * n_feat
            grad_b = 0.0
            for feats, label in train_examples_std:
                z = sum(w * f for w, f in zip(weights, feats)) + bias
                z = max(LOGISTIC_CLAMP[0], min(LOGISTIC_CLAMP[1], z))
                p = sigmoid(z)
                err = p - label
                for d in range(n_feat):
                    grad_w[d] += err * feats[d]
                grad_b += err
            for d in range(n_feat):
                grad_w[d] = grad_w[d] / n_ex + LOGISTIC_L2 * weights[d]
            grad_b = grad_b / n_ex
            for d in range(n_feat):
                weights[d] -= LOGISTIC_LR * grad_w[d]
            bias -= LOGISTIC_LR * grad_b

        enter_probs = {}
        for uid in candidates:
            r0 = last_seen_rank(uid)
            age = chart_age(uid)
            p_now = playing_at(uid, n_snap - 1)
            log10_play = math.log10(p_now) if p_now is not None and p_now > 0 else 0.0
            spk = spike_at(uid, n_snap - 1)
            p_prev3 = playing_at(uid, n_snap - 4)
            growth = p_now / p_prev3 if (p_now is not None and p_prev3 is not None and p_prev3 > 0) else 0.0
            r_prev3 = None
            if n_snap - 4 >= 0:
                for g in snapshots[n_snap - 4]["games"]:
                    if g["universe_id"] == uid:
                        r_prev3 = g["rank"]
                        break
            mom = (r_prev3 - r0) if r_prev3 is not None else 0
            feats = [float(r0), float(age), log10_play, float(spk), float(growth), float(mom)]
            feats = standardize(feats, means, stds)
            z = sum(w * f for w, f in zip(weights, feats)) + bias
            z = max(LOGISTIC_CLAMP[0], min(LOGISTIC_CLAMP[1], z))
            enter_probs[uid] = sigmoid(z)
        mean_enter = sum(enter_probs.values()) / len(enter_probs)
        for uid in candidates:
            if mean_enter > 0:
                factor = (enter_probs[uid] / mean_enter) ** 0.5
            else:
                factor = 1.0
            for k in range(1, 11):
                mixed[uid][k] *= factor

    for k in range(1, 11):
        tot = sum(mixed[uid][k] for uid in candidates)
        if tot > 0:
            for uid in candidates:
                mixed[uid][k] = mixed[uid][k] / tot * NAMED_MASS
        else:
            for uid in candidates:
                mixed[uid][k] = NAMED_MASS / len(candidates)

    # --- Step 9: final ---
    rows = []
    for uid in candidates:
        for k in range(1, 11):
            p = mixed[uid][k]
            if p >= MIN_P:
                rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    for k in range(1, 11):
        place_rows = [r for r in rows if r["place"] == k]
        place_rows.sort(key=lambda r: (-r["p"], r["universe_id"]))
        place_rows = place_rows[:100]
        rows = [r for r in rows if r not in place_rows]
        rows.extend(place_rows)

    top10 = sorted(candidates, key=lambda uid: expected_rank[uid])[:10]
    if len(top10) < 10:
        chart_games = sorted(last["games"], key=lambda g: g["rank"])
        for g in chart_games:
            if g["universe_id"] not in top10:
                top10.append(g["universe_id"])
            if len(top10) == 10:
                break

    # --- Step 10: dead-zone rows ---
    named_ids = {r["universe_id"] for r in rows}
    for g in last["games"]:
        uid = g["universe_id"]
        r = g["rank"]
        if 31 <= r <= 50 and uid not in named_ids:
            for k in range(1, 11):
                rows.append({"universe_id": uid, "place": k, "p": 0.0})
            named_ids.add(uid)

    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    rows = [r for r in rows if r["p"] >= MIN_P or r["p"] == 0.0]

    out = {"probabilities": rows, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "forecast.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
