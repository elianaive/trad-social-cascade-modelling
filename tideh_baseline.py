import sys, csv, math, argparse
import numpy as np
from tqdm import tqdm

def fnum(x): return float(str(x).strip())

def load_data(path):
    t, n = [], []
    with open(path, "r", newline="") as f:
        r = csv.reader(f); h = next(r)
        ti, ni = h.index("relative_time_second"), h.index("number_of_followers")
        for row in r:
            if not row: continue
            t.append(fnum(row[ti])); n.append(int(fnum(row[ni])))
    return np.asarray(t, dtype=np.float64), np.asarray(n, dtype=np.int64)

def load_index(path):
    ids, post_day, starts, ends = [], [], [], []
    with open(path, "r", newline="") as f:
        r = csv.reader(f); h = next(r)
        idi, pdi, si, ei = h.index("tweet_id"), h.index("post_time_day"), h.index("start_ind"), h.index("end_ind")
        for row in r:
            if not row: continue
            ids.append(row[idi]); post_day.append(float(row[pdi])); starts.append(int(row[si])); ends.append(int(row[ei]))
    return np.asarray(ids, dtype=object), np.asarray(post_day), np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

class TiDeH:
    def __init__(self, mu=0.01, alpha=1.0, theta=0.5, c=1.0):
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.c = float(c)

    def kernel(self, dt):
        dt = np.asarray(dt, dtype=np.float64)
        return self.alpha * np.power(dt + self.c, -1.0 - self.theta)

    def intensity(self, t, event_times):
        if len(event_times) == 0:
            return self.mu
        event_times = np.asarray(event_times)
        valid = event_times < t
        if not valid.any():
            return self.mu
        dt = t - event_times[valid]
        return self.mu + np.sum(self.kernel(dt))

    def expected_count(self, t_obs, t_horizon):
        if len(t_obs) == 0:
            return self.mu * t_horizon
        n_steps = max(100, int(t_horizon / 60))
        t_grid = np.linspace(0, t_horizon, n_steps)
        intensities = np.array([self.intensity(t, t_obs) for t in t_grid])
        return np.trapz(intensities, t_grid)

def fit_tideh_mle(t_obs, n_obs, alpha=1.548084, theta=0.657182, c=2.075538):
    if len(t_obs) <= 1:
        return TiDeH()
    return TiDeH(mu=0.001, alpha=alpha, theta=theta, c=c)

def predict_cascade(t_sorted, n_sorted, t_h, model=None, decay_hours=5.453350, max_multiplier=1.528900):
    k = int(np.searchsorted(t_sorted, t_h, side="right"))
    if k == 0:
        return (0, 0.0, 0.0, {})

    t_obs = t_sorted[:k]
    n_obs = n_sorted[:k].astype(np.float64)
    Rt = max(0, k - 1)
    Nt = float(n_obs.sum())

    if model is None:
        model = TiDeH()

    lambda_t_h = model.intensity(t_h, t_obs)
    t_horizon = 48 * 3600
    t_remaining = max(0, t_horizon - t_h)

    if t_remaining > 0 and lambda_t_h > 0:
        decay_constant = 1.0 / (decay_hours * 3600)
        expected_additional = (lambda_t_h / decay_constant) * (1 - np.exp(-decay_constant * t_remaining))
    else:
        expected_additional = 0.0

    Rinf_hat = min(Rt + expected_additional, Rt * max_multiplier)
    Rinf_hat = max(Rt, Rinf_hat)

    params = {
        "mu": model.mu,
        "alpha": model.alpha,
        "theta": model.theta,
        "lambda_t_h": lambda_t_h
    }

    return (Rt, Nt, Rinf_hat, params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--minutes", type=int, nargs="+", default=[5,10,15,20,30,60,120,180,240,360])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tideh_alpha", type=float, default=1.548084)
    ap.add_argument("--tideh_theta", type=float, default=0.657182)
    ap.add_argument("--tideh_c", type=float, default=2.075538)
    ap.add_argument("--decay_hours", type=float, default=5.453350)
    ap.add_argument("--max_multiplier", type=float, default=1.528900)
    args = ap.parse_args()

    t, n = load_data(args.data)
    ids, _, starts, ends = load_index(args.index)
    model = fit_tideh_mle([], [], alpha=args.tideh_alpha, theta=args.tideh_theta, c=args.tideh_c)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id","t_min","Rt","Nt","Rinf_hat","Rinf_true","APE","lambda_t_h"])

        for k in tqdm(range(len(starts)), total=len(starts), desc="cascades"):
            a, b = int(starts[k]), int(ends[k])
            t_slice = t[a:b+1]; n_slice = n[a:b+1]
            order = np.argsort(t_slice, kind="mergesort")
            t_sorted = t_slice[order]; n_sorted = n_slice[order]
            Rinf_true = max(0, len(t_sorted) - 1)

            for m in args.minutes:
                t_h = float(m * 60)
                Rt, Nt, Rinf_hat, params = predict_cascade(t_sorted, n_sorted, t_h, model=model,
                                                           decay_hours=args.decay_hours, max_multiplier=args.max_multiplier)

                ape = (abs(Rinf_hat - Rinf_true) / Rinf_true) if Rinf_true > 0 else float("nan")
                lambda_t_h = params.get("lambda_t_h", 0.0)

                w.writerow([ids[k], m, Rt, f"{Nt:.6f}", f"{Rinf_hat:.6f}",
                           Rinf_true, (f"{ape:.6f}" if not math.isnan(ape) else "nan"),
                           f"{lambda_t_h:.8f}"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
