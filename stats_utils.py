"""
stats_utils.py - statistical engine for the Customer Behaviour Lab dashboard.

Everything numerical lives here (no Streamlit code), so you can import it in a
notebook, unit-test it, or quote the formulas in your report.

Sections
  1. Loading / cleaning transactions
  2. Customer-level measures (spend, frequency, recency, state)
  3. Markov chain on customer states
  4. Distribution fitting (lognormal, Poisson/NegBin, exponential/Weibull)
  5. Inequality (Lorenz / Gini)
  6. Hypothesis tests (t, F, ANOVA + assumptions, post-hoc, effect sizes)
  7. What-if retention simulation
  8. Survival (Kaplan-Meier)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

STATES = ["Active", "Regular", "Inactive"]

# --------------------------------------------------------------------------
# 1. LOADING / CLEANING
# --------------------------------------------------------------------------
_ALIASES = {
    "invoice": ["InvoiceNo", "Invoice", "Order ID", "OrderID", "order_id", "invoice_no"],
    "customer": ["CustomerID", "Customer ID", "customer_id", "CustomerId"],
    "date": ["InvoiceDate", "Date", "Order Date", "order_date", "invoice_date"],
    "product": ["Description", "Product", "Product Name", "StockCode"],
    "qty": ["Quantity", "Qty", "quantity"],
    "price": ["UnitPrice", "Unit Price", "Price", "unit_price"],
    "country": ["Country", "country"],
}


def _key(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("_", "")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names (UCI Online Retail, Kaggle variants) to a fixed schema."""
    lookup = {_key(c): c for c in df.columns}
    rename = {}
    for canon, names in _ALIASES.items():
        for n in names:
            if _key(n) in lookup:
                rename[lookup[_key(n)]] = canon
                break
    out = df.rename(columns=rename)
    missing = [c for c in ["invoice", "customer", "date", "qty", "price"] if c not in out.columns]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}. "
            f"Your columns are: {list(df.columns)}"
        )
    if "country" not in out.columns:
        out["country"] = "Unknown"
    if "product" not in out.columns:
        out["product"] = "N/A"
    return out[["invoice", "customer", "date", "product", "qty", "price", "country"]].copy()


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop missing customers, cancellations/returns (qty<=0) and zero prices; add `amount`."""
    df = standardize_columns(raw)
    df["invoice"] = df["invoice"].astype(str)
    df = df.dropna(subset=["customer", "date"])
    df["customer"] = df["customer"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    df["amount"] = df["qty"] * df["price"]            # Quantity x Unit Price
    df["country"] = df["country"].fillna("Unknown").astype(str)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. CUSTOMER-LEVEL MEASURES
# --------------------------------------------------------------------------
def state_from_recency(recency, active_days: int, inactive_days: int):
    """Active: bought within `active_days`; Regular: within `inactive_days`; else Inactive."""
    recency = np.asarray(recency, dtype=float)
    return np.select([recency <= active_days, recency <= inactive_days],
                     ["Active", "Regular"], default="Inactive")


def customer_table(df: pd.DataFrame, snapshot: pd.Timestamp, active_days: int, inactive_days: int):
    g = df.groupby("customer")
    t = g.agg(
        total_spend=("amount", "sum"),
        n_orders=("invoice", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    t["country"] = df.groupby("customer")["country"].agg(lambda s: s.value_counts().index[0])
    t["recency"] = (snapshot - t["last"]).dt.days
    t["tenure_days"] = (snapshot - t["first"]).dt.days
    t["state"] = state_from_recency(t["recency"], active_days, inactive_days)
    return t


# --------------------------------------------------------------------------
# 3. MARKOV CHAIN
# --------------------------------------------------------------------------
def monthly_states(df: pd.DataFrame, active_days: int, inactive_days: int) -> pd.DataFrame:
    """
    Customer x month-end matrix of states (NaN before a customer's first purchase).
    State at each month-end is decided from recency measured at that month-end.
    """
    period = df["date"].dt.to_period("M")
    periods = pd.period_range(period.min(), period.max(), freq="M")
    last_in_month = df.groupby(["customer", period])["date"].max().unstack()
    last_in_month = last_in_month.reindex(columns=periods)
    last_seen = last_in_month.ffill(axis=1)           # latest purchase up to that month
    cols = {}
    for p in periods:
        rec = (p.end_time - last_seen[p]).dt.days
        st = pd.Series(state_from_recency(rec.fillna(0).values, active_days, inactive_days),
                       index=last_seen.index)
        cols[str(p)] = st.where(rec.notna())
    S = pd.DataFrame(cols)
    S.index.name = "customer"
    return S


def transition_counts(S: pd.DataFrame) -> pd.DataFrame:
    cols = list(S.columns)
    C = pd.DataFrame(0, index=STATES, columns=STATES, dtype=float)
    for a, b in zip(cols[:-1], cols[1:]):
        pair = pd.DataFrame({"from": S[a], "to": S[b]}).dropna()
        ct = pd.crosstab(pair["from"], pair["to"]).reindex(index=STATES, columns=STATES, fill_value=0)
        C = C + ct
    return C


def pair_counts(S: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    pair = pd.DataFrame({"from": S[a], "to": S[b]}).dropna()
    return pd.crosstab(pair["from"], pair["to"]).reindex(index=STATES, columns=STATES, fill_value=0)


def transition_matrix(C: pd.DataFrame) -> pd.DataFrame:
    P = C.div(C.sum(axis=1).replace(0, np.nan), axis=0)
    for s in STATES:                                   # unseen row -> stay put
        if P.loc[s].isna().all():
            P.loc[s] = 0.0
            P.loc[s, s] = 1.0
    return P


def steady_state(P: pd.DataFrame) -> pd.Series:
    """Solve pi P = pi, sum(pi)=1."""
    n = len(P)
    A = np.vstack([P.T.values - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0, None)
    return pd.Series(pi / pi.sum(), index=P.index)


def mean_first_passage(P: pd.DataFrame, target: str = "Inactive") -> pd.Series:
    """Expected number of months to first reach `target` from each other state."""
    keep = [s for s in P.index if s != target]
    Q = P.loc[keep, keep].values
    try:
        m = np.linalg.solve(np.eye(len(keep)) - Q, np.ones(len(keep)))
    except np.linalg.LinAlgError:
        m = np.full(len(keep), np.nan)
    return pd.Series(m, index=keep)


def markov_order_test(S: pd.DataFrame) -> dict:
    """
    Test the first-order Markov property.
    H0: next state is independent of the previous state, given the current state.
    (chi-square test of independence on (prev, next) within each current state, summed.)
    """
    cols = list(S.columns)
    chi_total, dof_total = 0.0, 0
    for s in STATES:
        frames = []
        for a, b, c in zip(cols[:-2], cols[1:-1], cols[2:]):
            trip = S[[a, b, c]].dropna()
            trip = trip[trip[b] == s]
            frames.append(pd.DataFrame({"prev": trip[a].values, "next": trip[c].values}))
        if not frames:
            continue
        pairs = pd.concat(frames, ignore_index=True)
        ct = pd.crosstab(pairs["prev"], pairs["next"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, _, dof, _ = stats.chi2_contingency(ct)
            chi_total += chi2
            dof_total += dof
    p = float(stats.chi2.sf(chi_total, dof_total)) if dof_total else np.nan
    return {"chi2": chi_total, "dof": dof_total, "p": p}


def state_monthly_spend(df: pd.DataFrame, S: pd.DataFrame) -> pd.Series:
    """Average spend in a month for a customer who is in each state at that month-end."""
    per = df["date"].dt.to_period("M").astype(str)
    sp = (df.groupby(["customer", per])["amount"].sum().unstack()
            .reindex(index=S.index, columns=S.columns).fillna(0))
    st_long = S.reset_index().melt(id_vars="customer", var_name="period", value_name="state").dropna()
    sp_long = sp.reset_index().melt(id_vars="customer", var_name="period", value_name="spend")
    m = st_long.merge(sp_long, on=["customer", "period"])
    return m.groupby("state")["spend"].mean().reindex(STATES).fillna(0.0)


# --------------------------------------------------------------------------
# 4. DISTRIBUTION FITTING
# --------------------------------------------------------------------------
def _aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik


def fit_lognormal(x) -> dict:
    x = np.asarray(x, float)
    x = x[x > 0]
    sigma, _, scale = stats.lognorm.fit(x, floc=0)
    ks = stats.kstest(x, "lognorm", args=(sigma, 0, scale))
    return {"mu": float(np.log(scale)), "sigma": float(sigma), "scale": float(scale),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "n": len(x)}


def compare_spend_models(x) -> pd.DataFrame:
    """AIC / BIC for candidate models of positive, right-skewed spend."""
    x = np.asarray(x, float)
    x = x[x > 0]
    n = len(x)
    rows = []
    for name, dist, kw in [("Lognormal", stats.lognorm, {"floc": 0}),
                           ("Gamma", stats.gamma, {"floc": 0}),
                           ("Weibull", stats.weibull_min, {"floc": 0}),
                           ("Normal", stats.norm, {})]:
        params = dist.fit(x, **kw)
        ll = float(np.sum(dist.logpdf(x, *params)))
        k = 2
        rows.append({"Model": name, "Log-likelihood": ll,
                     "AIC": _aic(ll, k), "BIC": k * np.log(n) - 2 * ll})
    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out["Delta AIC"] = out["AIC"] - out["AIC"].min()
    return out


def monthly_order_counts(df: pd.DataFrame) -> np.ndarray:
    """Orders per customer per month, including zero months after the first purchase."""
    per = df["date"].dt.to_period("M")
    periods = pd.period_range(per.min(), per.max(), freq="M")
    cnt = (df.groupby(["customer", per])["invoice"].nunique().unstack(fill_value=0)
             .reindex(columns=periods, fill_value=0))
    first = per.groupby(df["customer"]).min().reindex(cnt.index)
    first_idx = periods.get_indexer(first)
    mask = np.arange(len(periods))[None, :] >= first_idx[:, None]
    return cnt.values[mask].astype(int)


def poisson_vs_negbin(counts, kmax: int = 10) -> dict:
    c = np.asarray(counts, int)
    m, v = c.mean(), c.var(ddof=1)
    ll_p = float(stats.poisson.logpmf(c, m).sum())
    out = {"mean": float(m), "var": float(v), "dispersion": float(v / m) if m > 0 else np.nan,
           "aic_poisson": _aic(ll_p, 1), "n": len(c)}
    ks = np.arange(0, kmax + 1)
    out["k"] = ks
    out["empirical"] = np.array([(c == k).mean() for k in ks])
    out["poisson_pmf"] = stats.poisson.pmf(ks, m)
    if v > m:
        p = m / v
        r = m * p / (1 - p)
        ll_nb = float(stats.nbinom.logpmf(c, r, p).sum())
        out.update(aic_nb=_aic(ll_nb, 2), nb_r=float(r), nb_p=float(p),
                   nb_pmf=stats.nbinom.pmf(ks, r, p))
    else:
        out.update(aic_nb=np.nan, nb_r=np.nan, nb_p=np.nan, nb_pmf=np.full(len(ks), np.nan))
    # chi-square goodness of fit for Poisson (pool tail so expected >= 5)
    exp = len(c) * stats.poisson.pmf(np.arange(0, 6), m)
    obs = np.array([(c == k).sum() for k in range(0, 6)])
    exp = np.append(exp, len(c) - exp.sum())
    obs = np.append(obs, len(c) - obs.sum())
    keep = exp > 5
    if keep.sum() > 2:
        scale = obs[keep].sum() / exp[keep].sum()
        chi2 = float(((obs[keep] - exp[keep] * scale) ** 2 / (exp[keep] * scale)).sum())
        dof = int(keep.sum() - 2)
        out["chi2_p"] = float(stats.chi2.sf(chi2, dof))
        out["chi2"], out["chi2_dof"] = chi2, dof
    else:
        out["chi2_p"] = out["chi2"] = np.nan
        out["chi2_dof"] = 0
    return out


def interpurchase_gaps(df: pd.DataFrame) -> np.ndarray:
    d = (df.assign(day=df["date"].dt.normalize())
           .drop_duplicates(["customer", "day"])
           .sort_values(["customer", "day"]))
    gaps = d.groupby("customer")["day"].diff().dt.days.dropna()
    return gaps[gaps > 0].values.astype(float)


def fit_exponential_weibull(gaps) -> dict:
    g = np.asarray(gaps, float)
    scale = g.mean()                                   # MLE of exponential
    ks = stats.kstest(g, "expon", args=(0, scale))
    ll_e = float(stats.expon.logpdf(g, 0, scale).sum())
    c, _, wscale = stats.weibull_min.fit(g, floc=0)
    ll_w = float(stats.weibull_min.logpdf(g, c, 0, wscale).sum())
    # Memorylessness: P(T > s+t | T > s) vs P(T > t)
    s, t = float(np.median(g)), float(np.median(g))
    surv_s = (g > s).mean()
    cond = (g > s + t).mean() / surv_s if surv_s > 0 else np.nan
    return {"rate": float(1 / scale), "mean_gap": float(scale), "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue), "weibull_shape": float(c), "weibull_scale": float(wscale),
            "aic_exp": _aic(ll_e, 1), "aic_weibull": _aic(ll_w, 2), "n": len(g),
            "s": s, "t": t, "p_cond": float(cond), "p_uncond": float((g > t).mean())}


# --------------------------------------------------------------------------
# 5. INEQUALITY
# --------------------------------------------------------------------------
def lorenz_gini(x):
    x = np.sort(np.asarray(x, float))
    cum = np.cumsum(x)
    cum = np.insert(cum, 0, 0) / cum[-1]
    pop = np.linspace(0, 1, len(cum))
    _trap = getattr(np, 'trapezoid', None) or np.trapz
    gini = 1 - 2 * _trap(cum, pop)
    return pop, cum, float(gini)


# --------------------------------------------------------------------------
# 6. HYPOTHESIS TESTS
# --------------------------------------------------------------------------
def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def f_test(a, b) -> dict:
    """Two-sided variance-ratio F-test, H0: var(a) == var(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    F = a.var(ddof=1) / b.var(ddof=1)
    dfn, dfd = len(a) - 1, len(b) - 1
    p = 2 * min(stats.f.cdf(F, dfn, dfd), stats.f.sf(F, dfn, dfd))
    return {"F": float(F), "dfn": dfn, "dfd": dfd, "p": float(min(p, 1.0))}


def t_test(a, b) -> dict:
    """Welch two-sample t-test (does not assume equal variances) + pooled version."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = stats.ttest_ind(a, b, equal_var=False)
    pooled = stats.ttest_ind(a, b, equal_var=True)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"t": float(w.statistic), "p": float(w.pvalue), "t_pooled": float(pooled.statistic),
            "p_pooled": float(pooled.pvalue), "diff": float(diff),
            "ci": (float(diff - 1.96 * se), float(diff + 1.96 * se)), "d": cohens_d(a, b)}


def anova_suite(groups: dict) -> dict:
    """One-way ANOVA + assumption checks + Kruskal-Wallis + Tukey HSD + eta-squared."""
    arrays = [np.asarray(v, float) for v in groups.values()]
    names = list(groups.keys())
    f = stats.f_oneway(*arrays)
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
    sst = ((allv - grand) ** 2).sum()
    lev = stats.levene(*arrays)
    kw = stats.kruskal(*arrays)
    rng = np.random.default_rng(0)
    shapiro = {}
    for n, a in zip(names, arrays):
        samp = a if len(a) <= 5000 else rng.choice(a, 5000, replace=False)
        shapiro[n] = float(stats.shapiro(samp).pvalue) if len(samp) >= 3 else np.nan
    tukey_df = None
    try:
        res = stats.tukey_hsd(*arrays)
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rows.append({"Group A": names[i], "Group B": names[j],
                             "Mean diff (A-B)": float(arrays[i].mean() - arrays[j].mean()),
                             "p (Tukey)": float(res.pvalue[i, j])})
        tukey_df = pd.DataFrame(rows)
    except Exception:
        pass
    return {"F": float(f.statistic), "p": float(f.pvalue), "eta2": float(ssb / sst) if sst else np.nan,
            "levene_p": float(lev.pvalue), "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "shapiro": shapiro, "tukey": tukey_df, "df_between": len(arrays) - 1,
            "df_within": int(len(allv) - len(arrays))}


def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 1, level: float = 0.95):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------
# 7. WHAT-IF RETENTION SIMULATION
# --------------------------------------------------------------------------
def adjust_matrix(P: pd.DataFrame, reactivation_boost: float, churn_reduction: float) -> pd.DataFrame:
    """
    reactivation_boost : absolute increase in P(Inactive -> Regular)  (0..1)
    churn_reduction    : relative cut (0..1) in P(Active -> Inactive) and P(Regular -> Inactive),
                         probability mass is kept in the customer's current state.
    """
    P2 = P.copy().astype(float)
    take = min(reactivation_boost, P2.loc["Inactive", "Inactive"])
    P2.loc["Inactive", "Regular"] += take
    P2.loc["Inactive", "Inactive"] -= take
    for s in ["Active", "Regular"]:
        cut = P2.loc[s, "Inactive"] * churn_reduction
        P2.loc[s, "Inactive"] -= cut
        P2.loc[s, s] += cut
    return P2


def project(P: pd.DataFrame, start: pd.Series, months: int) -> pd.DataFrame:
    v = start.reindex(P.index).values.astype(float)
    rows = [v.copy()]
    for _ in range(months):
        v = v @ P.values
        rows.append(v.copy())
    return pd.DataFrame(rows, columns=P.index)


# --------------------------------------------------------------------------
# 8. SURVIVAL
# --------------------------------------------------------------------------
def kaplan_meier(durations, events) -> pd.DataFrame:
    d = pd.DataFrame({"t": np.asarray(durations, float), "e": np.asarray(events, int)})
    grp = d.groupby("t")["e"].agg(["size", "sum"]).sort_index()
    n_at_risk = len(d) - grp["size"].cumsum().shift(fill_value=0)
    surv = (1 - grp["sum"] / n_at_risk).cumprod()
    out = pd.DataFrame({"t": grp.index.values, "survival": surv.values})
    return pd.concat([pd.DataFrame({"t": [0.0], "survival": [1.0]}), out], ignore_index=True)


# --------------------------------------------------------------------------
# 9. BUSINESS INSIGHTS (plain-English findings generated from the data)
# --------------------------------------------------------------------------
SEGMENT_ORDER = ["VIP", "Loyal", "New / occasional", "At risk", "Lost"]

SEGMENT_RULES = {
    "VIP": "Top 10% by total spend and not Inactive",
    "Loyal": "Active with 3 or more orders",
    "New / occasional": "Active with 1-2 orders",
    "At risk": "Regular: has gone quiet but not yet lost",
    "Lost": "Inactive: no purchase for a long time",
}

SEGMENT_ACTIONS = {
    "VIP": "Protect and reward: early access, loyalty tier, personal service. Losing one of these costs the most.",
    "Loyal": "Keep them buying: points, bundles and cross-sell of higher-value products.",
    "New / occasional": "Turn into repeat buyers: welcome emails and a second-purchase offer.",
    "At risk": "Win back before they lapse: a reminder or small offer timed to their usual buying gap.",
    "Lost": "Cheap reactivation campaign; spend effort mainly on the previously high-spending ones.",
}


def assign_segments(cust: pd.DataFrame) -> np.ndarray:
    p90 = cust["total_spend"].quantile(0.9)
    return np.select(
        [(cust["total_spend"] >= p90) & (cust["state"] != "Inactive"),
         (cust["n_orders"] >= 3) & (cust["state"] == "Active"),
         cust["state"] == "Active",
         cust["state"] == "Regular"],
        ["VIP", "Loyal", "New / occasional", "At risk"], default="Lost")


def segment_summary(cust: pd.DataFrame) -> pd.DataFrame:
    g = cust.groupby("segment")
    t = pd.DataFrame({
        "Customers": g.size(),
        "Revenue": g["total_spend"].sum(),
        "Avg spend": g["total_spend"].mean(),
        "Avg orders": g["n_orders"].mean(),
        "Avg days since last order": g["recency"].mean(),
    }).reindex(SEGMENT_ORDER).fillna(0)
    t["% of customers"] = t["Customers"] / t["Customers"].sum() * 100
    t["% of revenue"] = t["Revenue"] / t["Revenue"].sum() * 100
    t["What the company can do"] = [SEGMENT_ACTIONS[s] for s in t.index]
    return t


def _order_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("invoice").agg(amount=("amount", "sum"), items=("qty", "sum"),
                                     date=("date", "min"))


def business_findings(df: pd.DataFrame, cust: pd.DataFrame, P: pd.DataFrame, cur: str = "") -> list:
    """List of dicts: title, says (what the data says), do (what the company can do)."""
    money = lambda v: f"{cur}{v:,.0f}"
    out = []
    n = len(cust)
    total = cust["total_spend"].sum()

    # 1. revenue concentration
    ranked = cust["total_spend"].sort_values(ascending=False)
    top10 = ranked.head(max(1, int(0.1 * n))).sum() / total
    top20 = ranked.head(max(1, int(0.2 * n))).sum() / total
    out.append({
        "title": ("A small group of customers carries the business" if top10 >= 0.4
                  else "Revenue is fairly evenly spread across customers"),
        "says": f"The top 10% of customers bring {top10:.0%} of revenue and the top 20% bring {top20:.0%}.",
        "do": ("Create a VIP / loyalty tier with early access and personal service, and track VIP "
               "activity weekly so a quiet VIP is noticed before they leave." if top10 >= 0.4 else
               "No single group dominates, so broad programmes (loyalty points, regular offers) make "
               "more sense than a VIP-only strategy.")})

    # 2. one-time buyers
    one = (cust["n_orders"] == 1).mean()
    out.append({
        "title": "Many customers buy only once" if one >= 0.4 else "Most customers come back, but some never do",
        "says": f"{one:.0%} of customers placed a single order; {1 - one:.0%} bought two or more times.",
        "do": "Run a welcome email series and a second-purchase offer within the first weeks. "
              "Turning one-time buyers into repeat buyers is usually the cheapest growth."})

    # 3. timing of repeat purchases
    gaps = interpurchase_gaps(df)
    if len(gaps) >= 20:
        med, p80 = np.median(gaps), np.percentile(gaps, 80)
        out.append({
            "title": "Repeat purchases happen within a predictable window",
            "says": f"Half of repeat purchases come within {med:.0f} days of the previous one, and 80% within {p80:.0f} days.",
            "do": f"Send a reminder or offer at about day {med:.0f} after an order, and a stronger win-back "
                  f"message around day {p80:.0f}, when most customers who will return already have."})

    # 4. churn and reactivation (Markov chain)
    lapse = 1 - P.loc["Active", "Active"]
    react = P.loc["Inactive", ["Active", "Regular"]].sum()
    mfp = mean_first_passage(P).get("Active", np.nan)
    says = (f"Each month about {lapse:.0%} of Active customers stop being Active, "
            f"and only {react:.1%} of Inactive customers come back.")
    if not np.isnan(mfp):
        says += f" On average a customer takes about {mfp:.1f} months to reach Inactive."
    out.append({
        "title": ("Lost customers rarely return on their own" if react < 0.10
                  else "Some lost customers do return"),
        "says": says,
        "do": "Do not wait for customers to come back. Act while they are still 'Regular' (the At-risk "
              "segment): that is usually far cheaper than reviving them after they are Inactive."})

    # 5. lost high-value customers
    med_spend = cust["total_spend"].median()
    lost_hv = cust[(cust["state"] == "Inactive") & (cust["total_spend"] >= med_spend)]
    if len(lost_hv):
        out.append({
            "title": "Valuable customers have already been lost",
            "says": f"{len(lost_hv):,} Inactive customers had above-median spend, together worth "
                    f"{money(lost_hv['total_spend'].sum())} in past purchases.",
            "do": "Send these customers a personal win-back message (what is new, a limited offer). "
                  "Even a small recovery rate pays back, because they have already shown they will spend."})

    # 6. seasonality and peak times
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    if df["date"].max().day < 28:
        monthly = monthly.iloc[:-1]                    # drop partial last month
    if len(monthly) >= 6:
        best, worst = monthly.idxmax(), monthly.idxmin()
        inv = df.drop_duplicates("invoice")
        wd = inv["date"].dt.day_name().value_counts()
        hr = inv["date"].dt.hour.value_counts()
        out.append({
            "title": "Demand has clear peaks",
            "says": f"{best.strftime('%B %Y')} is the strongest full month ({money(monthly.max())}) and "
                    f"{worst.strftime('%B %Y')} the weakest ({money(monthly.min())}). Most orders arrive on "
                    f"{wd.index[0]}s, around {hr.index[0]}:00.",
            "do": "Plan stock, staffing and campaigns before the peak months, schedule emails and ads for "
                  "the busiest day and hour, and run promotions in the weak months to even out sales."})

    # 7. basket size
    ot = _order_totals(df)
    out.append({
        "title": "Order size can be increased",
        "says": f"The average order is {money(ot['amount'].mean())} (median {money(ot['amount'].median())}) "
                f"with about {ot['items'].mean():.0f} items.",
        "do": f"Test a free-delivery or discount threshold just above the median order value "
              f"(around {money(ot['amount'].median() * 1.2)}) and suggest bundles to lift basket size."})

    # 8. geography
    by_c = df.groupby("country")["amount"].sum().sort_values(ascending=False)
    if len(by_c) > 1:
        share = by_c.iloc[0] / by_c.sum()
        cc = cust.groupby("country").agg(customers=("total_spend", "size"), avg=("total_spend", "mean"))
        cc = cc[cc["customers"] >= 20]
        extra = ""
        if len(cc) >= 2:
            b = cc["avg"].idxmax()
            extra = (f" Among regions with at least 20 customers, {b} has the highest average spend "
                     f"per customer ({money(cc.loc[b, 'avg'])}).")
        out.append({
            "title": "Sales depend heavily on one market" if share >= 0.5 else "Sales are spread across several markets",
            "says": f"{by_c.index[0]} produces {share:.0%} of revenue.{extra}",
            "do": "Protect the main market, and test small, targeted campaigns in the highest-spending "
                  "smaller regions before investing widely."})

    # 9. products
    if "product" in df.columns and df["product"].nunique() > 10:
        pr = df.groupby("product")["amount"].sum().sort_values(ascending=False)
        out.append({
            "title": "A few products drive sales",
            "says": f"The top 10 products bring {pr.head(10).sum() / pr.sum():.0%} of revenue; "
                    f"the best seller is '{pr.index[0]}'.",
            "do": "Never let the best sellers run out of stock, feature them in campaigns, and bundle "
                  "slower products with them."})
    return out


def opportunity_estimates(df: pd.DataFrame, cust: pd.DataFrame,
                          p_second: float, p_winback: float, p_reengage: float) -> pd.DataFrame:
    """
    Transparent 'what if' sizing. Every customer converted is assumed to place ONE extra order
    at the relevant average order value.
    """
    aov = _order_totals(df)["amount"].mean()
    one_all = cust[cust["n_orders"] == 1]
    med = cust["total_spend"].median()
    lost = cust[(cust["state"] == "Inactive") & (cust["total_spend"] >= med)]
    risk = cust[cust["segment"] == "At risk"]
    lost_aov = (lost["total_spend"] / lost["n_orders"]).mean() if len(lost) else 0.0
    risk_aov = (risk["total_spend"] / risk["n_orders"]).mean() if len(risk) else 0.0
    rows = [
        ("Second purchase from one-time buyers", len(one_all), p_second, len(one_all) * p_second * aov),
        ("Win back lost high-value customers", len(lost), p_winback, len(lost) * p_winback * lost_aov),
        ("Re-engage at-risk customers", len(risk), p_reengage, len(risk) * p_reengage * risk_aov),
    ]
    out = pd.DataFrame(rows, columns=["Opportunity", "Customers in group", "Assumed success rate", "Extra revenue"])
    return out.sort_values("Extra revenue", ascending=False).reset_index(drop=True)


def insights_report(findings: list, seg: pd.DataFrame, opp: pd.DataFrame, cur: str = "") -> str:
    lines = ["# Customer Insights Report", "", "## What the data says and what to do", ""]
    for i, f in enumerate(findings, 1):
        lines += [f"### {i}. {f['title']}", f"- **Data:** {f['says']}", f"- **Action:** {f['do']}", ""]
    lines += ["## Customer segments", "", "| Segment | Customers | % revenue | Action |", "|---|---|---|---|"]
    for name, r in seg.iterrows():
        lines.append(f"| {name} | {int(r['Customers']):,} | {r['% of revenue']:.0f}% | {r['What the company can do']} |")
    lines += ["", "## Illustrative revenue opportunities", "",
              "(Assumes each converted customer places one extra order at the average order value.)", "",
              "| Opportunity | Customers | Success rate | Extra revenue |", "|---|---|---|---|"]
    for _, r in opp.iterrows():
        lines.append(f"| {r['Opportunity']} | {int(r['Customers in group']):,} | "
                     f"{r['Assumed success rate']:.0%} | {cur}{r['Extra revenue']:,.0f} |")
    return "\n".join(lines)
