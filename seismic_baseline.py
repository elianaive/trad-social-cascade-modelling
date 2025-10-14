import sys, csv, math, argparse
import numpy as np
from tqdm import tqdm

ALPHA = {5:0.389,10:0.803,15:0.772,20:0.709,30:0.680,60:0.562,120:0.454,180:0.378,240:0.352,360:0.326}

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

class Seismic:
    def __init__(self, s0=394.105631, theta=0.187739, c=None):
        self.s0 = float(s0); self.theta = float(theta)
        self.c = (1.0/(self.s0*(1.0+1.0/self.theta))) if c is None else float(c)
        self.s0_pow = self.s0**(1.0-self.theta)
        self.c_s0_sq_half = 0.5*self.c*self.s0*self.s0
        self.c_s0_theta = self.c*(self.s0**(1.0+self.theta))
        self.inv_theta = 1.0/self.theta; self.inv_1mtheta = 1.0/(1.0-self.theta)
    def A(self, u):
        u = np.asarray(u, dtype=np.float64); out = np.zeros_like(u)
        m1 = (u>0) & (u<=self.s0)
        if m1.any(): out[m1] = self.c*u[m1]
        m2 = u>self.s0
        if m2.any():
            um = u[m2]
            out[m2] = self.c*self.s0 + self.c*self.s0*self.inv_theta*(1.0 - (self.s0/um)**self.theta)
        return out
    def B(self, u):
        u = np.asarray(u, dtype=np.float64); out = np.zeros_like(u)
        m1 = (u>0) & (u<=self.s0)
        if m1.any(): out[m1] = 0.5*self.c*u[m1]*u[m1]
        m2 = u>self.s0
        if m2.any():
            um = u[m2]
            out[m2] = self.c_s0_sq_half + self.c_s0_theta*self.inv_1mtheta*(um**(1.0-self.theta) - self.s0_pow)
        return out
    def Kphi_integral(self, t_h, delta):
        delta = np.asarray(delta, dtype=np.float64)
        return (1.0 - delta/t_h)*self.A(delta) + (1.0/t_h)*self.B(delta)

def predict_cascade(t_sorted, n_sorted, t_h, model):
    k = int(np.searchsorted(t_sorted, t_h, side="right"))
    if k == 0: return (0,0.0,0.0,0.0,0.0,0.0,True)
    t_obs = t_sorted[:k]; n_obs = n_sorted[:k].astype(np.float64)
    Rt = max(0, k-1)
    N_t = float(n_obs.sum())
    Delta = t_h - t_obs
    Ne_t = float(model.A(Delta).dot(n_obs))
    if Rt>0:
        t_ret = t_obs[1:]
        tilde_R = float(t_ret.sum()/t_h)
    else:
        tilde_R = 0.0
    tilde_Ne = float(model.Kphi_integral(t_h, Delta).dot(n_obs))
    p_hat = (tilde_R/tilde_Ne) if tilde_Ne>0 else 0.0
    D = 1.0 - 20.0*p_hat
    unstable = (D<=0.0)
    return (Rt, N_t, Ne_t, tilde_R, tilde_Ne, p_hat, unstable)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--minutes", type=int, nargs="+", default=[5,10,15,20,30,60,120,180,240,360])
    ap.add_argument("--out", required=True)
    ap.add_argument("--s0", type=float, default=394.105631)
    ap.add_argument("--theta", type=float, default=0.187739)
    args = ap.parse_args()
    for m in args.minutes:
        if m not in ALPHA: sys.exit(2)
    t, n = load_data(args.data)
    ids, _, starts, ends = load_index(args.index)
    model = Seismic(s0=args.s0, theta=args.theta)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id","t_min","Rt","Nt","Ne_t","tilde_R","tilde_Ne","p_hat","Rinf_hat","Rinf_true","APE","unstable_flag"])
        for k in tqdm(range(len(starts)), total=len(starts), desc="cascades"):
            a, b = int(starts[k]), int(ends[k])
            t_slice = t[a:b+1]; n_slice = n[a:b+1]
            order = np.argsort(t_slice, kind="mergesort")
            t_sorted = t_slice[order]; n_sorted = n_slice[order]
            Rinf_true = max(0, len(t_sorted)-1)
            for m in args.minutes:
                t_h = float(m*60)
                Rt, Nt, Ne_t, tR, tNe, p_hat, unstable = predict_cascade(t_sorted, n_sorted, t_h, model)
                if unstable:
                    Rinf_hat = float("nan"); ape = float("nan")
                else:
                    Rinf_hat = Rt + ALPHA[m] * (p_hat * (Nt - Ne_t)) / (1.0 - 20.0*p_hat)
                    ape = (abs(Rinf_hat - Rinf_true) / Rinf_true) if Rinf_true>0 else float("nan")
                w.writerow([ids[k], m, Rt, f"{Nt:.6f}", f"{Ne_t:.6f}", f"{tR:.6f}", f"{tNe:.6f}", f"{p_hat:.8f}",
                            (f"{Rinf_hat:.6f}" if (not math.isnan(Rinf_hat)) else "nan"),
                            Rinf_true, (f"{ape:.6f}" if (not math.isnan(ape)) else "nan"), int(unstable)])
    return 0

if __name__ == "__main__":
    sys.exit(main())
