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

def predict_linear(t_sorted, n_sorted, t_h, total_window=48.0, decay_rate=0.1, max_multiplier=50):
    k = int(np.searchsorted(t_sorted, t_h, side="right"))
    if k == 0:
        return (0, 0.0, 0.0, {})

    t_obs = t_sorted[:k]
    n_obs = n_sorted[:k].astype(np.float64)
    Rt = max(0, k - 1)
    Nt = float(n_obs.sum())

    if Rt == 0:
        return (Rt, Nt, 0.0, {"growth_rate": 0.0})

    time_span = t_obs[-1] - t_obs[0]
    if time_span < 1.0:
        time_span = 1.0

    log_growth = math.log(Rt + 1) / (time_span / 3600.0)
    time_elapsed = t_h / 3600.0
    time_remaining = max(0.0, total_window - time_elapsed)

    predicted_log_growth = log_growth * math.exp(-decay_rate * time_elapsed)
    predicted_additional = predicted_log_growth * time_remaining

    Rinf_hat = (Rt + 1) * math.exp(predicted_additional) - 1
    Rinf_hat = max(Rt, Rinf_hat)

    max_size = Rt * max_multiplier
    Rinf_hat = min(Rinf_hat, max_size)

    params = {
        "growth_rate": log_growth,
        "predicted_log_growth": predicted_log_growth,
        "time_remaining": time_remaining
    }

    return (Rt, Nt, Rinf_hat, params)

def predict_log_linear(t_sorted, n_sorted, t_h, total_window=48.0, log_cap=20.0):
    k = int(np.searchsorted(t_sorted, t_h, side="right"))
    if k <= 1:
        return (0, 0.0, 0.0, {})

    t_obs = t_sorted[:k]
    Rt = max(0, k - 1)
    Nt = float(n_sorted[:k].sum())

    cumulative_counts = np.arange(1, len(t_obs) + 1)
    log_counts = np.log(cumulative_counts)

    if len(t_obs) >= 3:
        coeffs = np.polyfit(t_obs, log_counts, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        t_final = total_window * 3600
        log_final = slope * t_final + intercept
        log_final = min(log_final, log_cap)
        Rinf_hat = math.exp(log_final)
        Rinf_hat = max(Rt, min(Rinf_hat, Rt * 100))
    else:
        Rinf_hat = Rt * 2

    params = {"slope": slope if len(t_obs) >= 3 else 0.0}
    return (Rt, Nt, Rinf_hat, params)

def predict_cascade(t_sorted, n_sorted, t_h, method="linear", **kwargs):
    if method == "log_linear":
        return predict_log_linear(t_sorted, n_sorted, t_h, **kwargs)
    else:
        return predict_linear(t_sorted, n_sorted, t_h, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--minutes", type=int, nargs="+", default=[5,10,15,20,30,60,120,180,240,360])
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["linear", "log_linear"], default="log_linear")
    ap.add_argument("--total_window", type=float, default=48.0)
    ap.add_argument("--decay_rate", type=float, default=0.1)
    ap.add_argument("--max_multiplier", type=float, default=50.0)
    ap.add_argument("--log_cap", type=float, default=20.0)
    args = ap.parse_args()

    t, n = load_data(args.data)
    ids, _, starts, ends = load_index(args.index)

    kwargs = {
        "total_window": args.total_window,
        "decay_rate": args.decay_rate,
        "max_multiplier": args.max_multiplier,
        "log_cap": args.log_cap
    }

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id","t_min","Rt","Nt","Rinf_hat","Rinf_true","APE","method"])

        for k in tqdm(range(len(starts)), total=len(starts), desc="cascades"):
            a, b = int(starts[k]), int(ends[k])
            t_slice = t[a:b+1]; n_slice = n[a:b+1]
            order = np.argsort(t_slice, kind="mergesort")
            t_sorted = t_slice[order]; n_sorted = n_slice[order]
            Rinf_true = max(0, len(t_sorted) - 1)

            for m in args.minutes:
                t_h = float(m * 60)
                Rt, Nt, Rinf_hat, params = predict_cascade(t_sorted, n_sorted, t_h, method=args.method, **kwargs)

                ape = (abs(Rinf_hat - Rinf_true) / Rinf_true) if Rinf_true > 0 else float("nan")

                w.writerow([ids[k], m, Rt, f"{Nt:.6f}", f"{Rinf_hat:.6f}",
                           Rinf_true, (f"{ape:.6f}" if not math.isnan(ape) else "nan"),
                           args.method])
    return 0

if __name__ == "__main__":
    sys.exit(main())
