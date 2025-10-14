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

class Hawkes:
    def __init__(self, mu=0.01, alpha=1.0, beta=2.0):
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @property
    def branching_ratio(self):
        return self.alpha / self.beta if self.beta > 0 else float('inf')

    def kernel(self, dt):
        dt = np.asarray(dt, dtype=np.float64)
        return self.alpha * np.exp(-self.beta * dt)

    def intensity(self, t, event_times, follower_counts=None):
        if len(event_times) == 0:
            return self.mu

        event_times = np.asarray(event_times)
        valid = event_times < t
        if not valid.any():
            return self.mu

        dt = t - event_times[valid]
        kernel_vals = self.kernel(dt)

        if follower_counts is not None:
            follower_counts = np.asarray(follower_counts)
            weights = follower_counts[valid] / np.mean(follower_counts[valid])
            kernel_vals = kernel_vals * weights

        return self.mu + np.sum(kernel_vals)

    def expected_total_events(self, n_obs):
        n = self.branching_ratio
        if n >= 1.0:
            return float('inf')
        return n_obs / (1.0 - n)

def fit_hawkes_simple(t_obs, n_obs, beta_min=0.000068, beta_max=0.028621, alpha_factor=0.426937):
    if len(t_obs) <= 2:
        return Hawkes()

    inter_times = np.diff(t_obs)
    mean_inter = np.mean(inter_times)

    beta = 1.0 / (mean_inter + 1e-6)
    beta = np.clip(beta, beta_min, beta_max)

    if len(t_obs) > 3:
        early_rate = len(t_obs) / (t_obs[-1] - t_obs[0] + 1e-6)
        alpha = early_rate * 0.5
        alpha = np.clip(alpha, 0.1, beta * alpha_factor)
    else:
        alpha = beta * 0.5

    mu = 0.001
    return Hawkes(mu=mu, alpha=alpha, beta=beta)

def predict_cascade(t_sorted, n_sorted, t_h, model=None, br_threshold=0.938374, max_multiplier=18.040461):
    k = int(np.searchsorted(t_sorted, t_h, side="right"))
    if k == 0:
        return (0, 0.0, 0.0, {})

    t_obs = t_sorted[:k]
    n_obs = n_sorted[:k].astype(np.float64)
    Rt = max(0, k - 1)
    Nt = float(n_obs.sum())

    if model is None:
        model = Hawkes()

    br = model.branching_ratio

    if br >= br_threshold:
        Rinf_hat = Rt * 3
    elif br > 0:
        expected_total = Rt / max(1.0 - br, 0.1)
        Rinf_hat = min(expected_total, Rt * 5)
    else:
        Rinf_hat = Rt * 1.5

    Rinf_hat = max(Rt, min(Rinf_hat, Rt * max_multiplier))

    params = {
        "mu": model.mu,
        "alpha": model.alpha,
        "beta": model.beta,
        "branching_ratio": br,
        "lambda_t_h": model.intensity(t_h, t_obs, n_obs)
    }

    return (Rt, Nt, Rinf_hat, params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--minutes", type=int, nargs="+", default=[5,10,15,20,30,60,120,180,240,360])
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta_min", type=float, default=0.000068)
    ap.add_argument("--beta_max", type=float, default=0.028621)
    ap.add_argument("--alpha_factor", type=float, default=0.426937)
    ap.add_argument("--br_threshold", type=float, default=0.938374)
    ap.add_argument("--max_multiplier", type=float, default=18.040461)
    args = ap.parse_args()

    t, n = load_data(args.data)
    ids, _, starts, ends = load_index(args.index)
    model = fit_hawkes_simple([], [], beta_min=args.beta_min, beta_max=args.beta_max, alpha_factor=args.alpha_factor)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id","t_min","Rt","Nt","Rinf_hat","Rinf_true","APE","branching_ratio","lambda_t_h"])

        for k in tqdm(range(len(starts)), total=len(starts), desc="cascades"):
            a, b = int(starts[k]), int(ends[k])
            t_slice = t[a:b+1]; n_slice = n[a:b+1]
            order = np.argsort(t_slice, kind="mergesort")
            t_sorted = t_slice[order]; n_sorted = n_slice[order]
            Rinf_true = max(0, len(t_sorted) - 1)

            for m in args.minutes:
                t_h = float(m * 60)
                Rt, Nt, Rinf_hat, params = predict_cascade(t_sorted, n_sorted, t_h, model=model,
                                                           br_threshold=args.br_threshold, max_multiplier=args.max_multiplier)

                ape = (abs(Rinf_hat - Rinf_true) / Rinf_true) if Rinf_true > 0 else float("nan")
                br = params.get("branching_ratio", 0.0)
                lambda_t_h = params.get("lambda_t_h", 0.0)

                w.writerow([ids[k], m, Rt, f"{Nt:.6f}", f"{Rinf_hat:.6f}",
                           Rinf_true, (f"{ape:.6f}" if not math.isnan(ape) else "nan"),
                           f"{br:.6f}", f"{lambda_t_h:.8f}"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
