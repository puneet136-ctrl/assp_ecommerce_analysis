"""
Customer Insights Studio - a business-first view of e-commerce customer data.
Run with:  streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import stats_utils as su
from generate_sample_data import make_sample

st.set_page_config(page_title="Customer Insights Studio", page_icon="📈", layout="wide")

SEG_COLORS = {"VIP": "#6a3fc4", "Loyal": "#2e9e6b", "New / occasional": "#3b8fd4",
              "At risk": "#f0a030", "Lost": "#d9534f"}


# ----------------------------------------------------------------- helpers
def show(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner="Generating sample data…")
def load_sample():
    return make_sample()


@st.cache_data(show_spinner="Reading file…")
def read_upload(name, content):
    bio = io.BytesIO(content)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    try:
        return pd.read_csv(bio, encoding="utf-8")
    except UnicodeDecodeError:
        bio.seek(0)
        return pd.read_csv(bio, encoding="ISO-8859-1")


@st.cache_data(show_spinner="Cleaning transactions…")
def prepare(raw):
    return su.clean_transactions(raw)


@st.cache_data(show_spinner="Analysing customers…")
def build_model(df, active_days, lost_days):
    snapshot = df["date"].max() + pd.Timedelta(days=1)
    cust = su.customer_table(df, snapshot, active_days, lost_days)
    cust["segment"] = su.assign_segments(cust)
    S = su.monthly_states(df, active_days, lost_days)
    P = su.transition_matrix(su.transition_counts(S))
    return cust, S, P


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📈 Customer Insights Studio")
    page = st.radio("Go to", ["📊 Business snapshot", "👥 Customer segments", "💡 Insights & actions",
                              "💰 Revenue opportunities", "📣 Campaign planner", "🔎 Customer 360",
                              "🧮 Statistics behind it"])
    st.divider()
    src = st.radio("Data source", ["Built-in sample data", "Upload my dataset"])
    upload = None
    if src == "Upload my dataset":
        upload = st.file_uploader("CSV or Excel (Online Retail format)", type=["csv", "xlsx", "xls"])
    st.caption("Business rules")
    active_days = st.slider("Active = bought within (days)", 7, 90, 30)
    lost_days = st.slider("Lost = no purchase for more than (days)", active_days + 30, 365,
                          max(90, active_days + 30))
    cur = st.selectbox("Currency symbol", ["£", "₹", "$", "€", ""])

if src == "Upload my dataset":
    if upload is None:
        st.info("Upload a dataset in the sidebar, or switch to the built-in sample data.")
        st.stop()
    raw = read_upload(upload.name, upload.getvalue())
else:
    raw = load_sample()

try:
    df = prepare(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()
if df["date"].dt.to_period("M").nunique() < 4:
    st.error("Need at least 4 months of data.")
    st.stop()

cust, S, P = build_model(df, active_days, lost_days)
money = lambda v: f"{cur}{v:,.0f}"
n_orders = df["invoice"].nunique()
revenue = cust["total_spend"].sum()
aov = revenue / n_orders

st.caption(f"Data: {len(df):,} transactions · {len(cust):,} customers · "
           f"{df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")

# =================================================================== SNAPSHOT
if page.startswith("📊"):
    st.title("Business snapshot")
    c = st.columns(5)
    c[0].metric("Customers", f"{len(cust):,}")
    c[1].metric("Revenue", money(revenue))
    c[2].metric("Average order", money(aov))
    c[3].metric("Repeat customers", f"{(cust['n_orders'] >= 2).mean():.0%}")
    c[4].metric("Lost customers", f"{(cust['state'] == 'Inactive').mean():.0%}")

    l, r = st.columns([3, 2])
    with l:
        mon = df.groupby(df["date"].dt.to_period("M").astype(str))["amount"].sum().reset_index()
        mon.columns = ["Month", "Revenue"]
        show(px.bar(mon, x="Month", y="Revenue", title="Revenue by month"))
    with r:
        sc = cust["segment"].value_counts().reindex(su.SEGMENT_ORDER).fillna(0).reset_index()
        sc.columns = ["Segment", "Customers"]
        show(px.pie(sc, names="Segment", values="Customers", hole=0.5, color="Segment",
                    color_discrete_map=SEG_COLORS, title="Customer base by segment"))

    st.subheader("Health check")
    ranked = cust["total_spend"].sort_values(ascending=False)
    top10 = ranked.head(max(1, int(0.1 * len(cust)))).sum() / revenue
    lost_share = (cust["segment"] == "Lost").mean()
    lost_rev = cust.loc[cust["segment"] == "Lost", "total_spend"].sum() / revenue
    st.info(f"💎 The top 10% of customers bring **{top10:.0%}** of revenue.")
    st.warning(f"⚠️ **{lost_share:.0%}** of customers are Lost; they account for **{lost_rev:.0%}** of "
               "all revenue ever earned, so there is real money to win back.")
    st.success(f"🛒 The average order is **{money(aov)}**; "
               f"**{(cust['n_orders'] == 1).mean():.0%}** of customers have ordered only once.")

# =================================================================== SEGMENTS
elif page.startswith("👥"):
    st.title("Customer segments")
    seg = su.segment_summary(cust)
    cols = st.columns(5)
    for col, name in zip(cols, su.SEGMENT_ORDER):
        row = seg.loc[name]
        col.metric(name, f"{int(row['Customers']):,}", f"{row['% of revenue']:.0f}% of revenue", delta_color="off")
    fig = go.Figure()
    fig.add_bar(x=seg.index, y=seg["% of customers"], name="% of customers", marker_color="#9db8ef")
    fig.add_bar(x=seg.index, y=seg["% of revenue"], name="% of revenue", marker_color="#2e9e6b")
    fig.update_layout(barmode="group", title="Share of customers vs share of revenue", yaxis_title="Percent")
    show(fig)

    st.subheader("What each segment means and what to do")
    guide = pd.DataFrame({
        "Segment": su.SEGMENT_ORDER,
        "Who they are": [su.SEGMENT_RULES[s] for s in su.SEGMENT_ORDER],
        "Customers": [int(seg.loc[s, "Customers"]) for s in su.SEGMENT_ORDER],
        "Avg spend": [round(seg.loc[s, "Avg spend"], 1) for s in su.SEGMENT_ORDER],
        "What the company can do": [su.SEGMENT_ACTIONS[s] for s in su.SEGMENT_ORDER],
    })
    table(guide, hide_index=True)

    st.subheader("Campaign list: who exactly should we contact?")
    pick = st.selectbox("Segment", su.SEGMENT_ORDER, index=3)
    lst = (cust[cust["segment"] == pick]
           .sort_values("total_spend", ascending=False)
           [["total_spend", "n_orders", "recency", "country"]]
           .rename(columns={"total_spend": "Total spend", "n_orders": "Orders",
                            "recency": "Days since last order", "country": "Country"}))
    st.write(f"**{len(lst):,}** customers in **{pick}**, sorted by total spend (highest first).")
    table(lst.head(50).round(1))
    st.download_button(f"⬇️ Download full {pick} list (CSV)", lst.to_csv().encode(),
                       f"{pick.lower().replace(' / ', '_').replace(' ', '_')}_customers.csv", "text/csv")

# =================================================================== INSIGHTS
elif page.startswith("💡"):
    st.title("Insights and recommended actions")
    findings = su.business_findings(df, cust, P, cur)
    for i, f in enumerate(findings, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {f['title']}**")
            st.markdown(f"📌 **What the data says:** {f['says']}")
            st.markdown(f"✅ **What the company can do:** {f['do']}")
    with st.expander("Which statistical idea sits behind each finding?"):
        table(pd.DataFrame({
            "Finding": ["Revenue concentration", "One-time vs repeat buyers", "Time between purchases",
                        "Monthly lapse and return rates", "Segments and customer risk",
                        "Spend differs by customer group"],
            "Statistical idea": ["Lognormal distribution, Lorenz curve and Gini coefficient",
                                 "Poisson vs negative binomial counts (overdispersion)",
                                 "Exponential vs Weibull waiting times",
                                 "Markov chain: transition matrix, steady state",
                                 "Markov chain n-step probabilities P^n",
                                 "t-test, F-test and one-way ANOVA"]}), hide_index=True)
        st.caption("Open the 'Statistics behind it' page for the numbers and formulas.")
    seg = su.segment_summary(cust)
    opp = su.opportunity_estimates(df, cust, 0.10, 0.05, 0.10)
    st.download_button("⬇️ Download as a report (Markdown)",
                       su.insights_report(findings, seg, opp, cur).encode(),
                       "customer_insights_report.md", "text/markdown")

# =================================================================== OPPORTUNITIES
elif page.startswith("💰"):
    st.title("Revenue opportunities")
    st.caption("Illustrative sizing: each converted customer is assumed to place one extra order at the "
               "relevant average order value. Change the success rates to see the effect.")
    a, b, c = st.columns(3)
    p1 = a.slider("Second purchase from one-time buyers (%)", 0, 30, 10) / 100
    p2 = b.slider("Win-back of lost high-value customers (%)", 0, 30, 5) / 100
    p3 = c.slider("Re-engage at-risk customers (%)", 0, 30, 10) / 100
    opp = su.opportunity_estimates(df, cust, p1, p2, p3)
    fig = px.bar(opp, x="Extra revenue", y="Opportunity", orientation="h", text_auto=",.0f",
                 title="Estimated extra revenue by action")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    show(fig)
    gain = opp["Extra revenue"].sum()
    m = st.columns(2)
    m[0].metric("Total estimated extra revenue", money(gain))
    m[1].metric("As % of revenue so far", f"{gain / revenue:.1%}")
    show_opp = opp.copy()
    show_opp["Assumed success rate"] = (show_opp["Assumed success rate"] * 100).round(0).astype(int).astype(str) + "%"
    show_opp["Extra revenue"] = show_opp["Extra revenue"].round(0)
    table(show_opp, hide_index=True)

# =================================================================== CAMPAIGN PLANNER
elif page.startswith("📣"):
    st.title("Campaign planner")
    st.caption("Estimate whether a campaign would pay for itself before spending money. "
               "All inputs are assumptions you can change.")
    seg = cust.groupby("segment").agg(n=("total_spend", "size"), spend=("total_spend", "sum"),
                                      orders=("n_orders", "sum"))
    pick = st.selectbox("Target segment", su.SEGMENT_ORDER, index=3)
    n_seg = int(seg.loc[pick, "n"]) if pick in seg.index else 0
    seg_aov = float(seg.loc[pick, "spend"] / seg.loc[pick, "orders"]) if n_seg else 0.0
    c1, c2, c3 = st.columns(3)
    n_target = c1.number_input("Customers to contact", 0, max(n_seg, 1), n_seg)
    cost_each = c2.number_input(f"Cost per customer ({cur or 'currency'})", 0.0, 1000.0, 0.5, 0.1)
    resp = c3.slider("Customers who respond (%)", 1, 30, 5)
    c4, c5 = st.columns(2)
    order_val = c4.number_input(f"Average order value ({cur or 'currency'})", 0.0, 100000.0, round(seg_aov, 1), 1.0)
    margin = c5.slider("Profit margin on sales (%)", 5, 80, 30)

    responders = n_target * resp / 100
    rev = responders * order_val
    profit = rev * margin / 100
    cost = n_target * cost_each
    net = profit - cost
    roi = net / cost if cost > 0 else np.nan
    m = st.columns(4)
    m[0].metric("Campaign cost", money(cost))
    m[1].metric("Extra revenue", money(rev))
    m[2].metric("Net profit", money(net))
    m[3].metric("Return on spend", f"{roi:.0%}" if not np.isnan(roi) else "n/a")
    if cost > 0:
        be = cost_each / (order_val * margin / 100) * 100 if order_val > 0 else np.nan
        if net > 0:
            st.success(f"✅ Under these assumptions the campaign pays for itself. "
                       f"It would break even at about **{be:.1f}%** of customers responding.")
        else:
            st.error(f"❌ Under these assumptions the campaign loses money. "
                     f"It needs about **{be:.1f}%** of customers to respond to break even.")
    fig = go.Figure(go.Bar(x=["Cost", "Profit from sales", "Net result"], y=[cost, profit, net],
                           marker_color=["#d9534f", "#3b8fd4", "#2e9e6b" if net >= 0 else "#d9534f"]))
    fig.update_layout(title="Campaign economics")
    show(fig)

# =================================================================== STATISTICS
elif page.startswith("🧮"):
    st.title("Statistics behind the insights")
    st.caption("The ASSP methods that produce the business findings. Each block ends with what it means "
               "for the company.")
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)

    # ---- 1 spend ------------------------------------------------------
    st.subheader("1 · Customer spend follows a lognormal distribution")
    x = cust["total_spend"].values
    fit = su.fit_lognormal(x)
    pop, cum, gini = su.lorenz_gini(x)
    hi = np.percentile(x, 99)
    xs = np.linspace(x.min(), hi, 400)
    l, r = st.columns([3, 2])
    with l:
        fig = go.Figure()
        fig.add_histogram(x=x[x <= hi], histnorm="probability density", nbinsx=60, name="Observed",
                          marker_color="#9db8ef")
        fig.add_scatter(x=xs, y=stats.lognorm.pdf(xs, fit["sigma"], 0, fit["scale"]), name="Lognormal fit",
                        line=dict(color="#d9534f", width=3))
        fig.update_layout(title="Total spend per customer (clipped at 99th percentile)",
                          xaxis_title="Total spend", yaxis_title="Density")
        show(fig)
    with r:
        m = st.columns(2)
        m[0].metric("μ (log scale)", f"{fit['mu']:.2f}")
        m[1].metric("σ (log scale)", f"{fit['sigma']:.2f}")
        m = st.columns(2)
        m[0].metric("KS statistic", f"{fit['ks_stat']:.3f}")
        m[1].metric("Gini coefficient", f"{gini:.2f}")
        st.latex(r"\ln X \sim \mathcal{N}(\mu,\sigma^2)")
        st.latex(r"G = 1 - 2\int_0^1 L(p)\,dp")
    st.info(f"**So what:** spend is heavily right-skewed (Gini {gini:.2f}), so a few big customers carry "
            "the business. This is why a VIP programme matters.")

    # ---- 2 frequency --------------------------------------------------
    st.subheader("2 · Order counts: Poisson or negative binomial?")
    pn = su.poisson_vs_negbin(su.monthly_order_counts(df))
    l, r = st.columns([3, 2])
    with l:
        fig = go.Figure()
        fig.add_bar(x=pn["k"], y=pn["empirical"], name="Observed", marker_color="#9db8ef")
        fig.add_scatter(x=pn["k"], y=pn["poisson_pmf"], name="Poisson", mode="lines+markers",
                        line=dict(color="#d9534f"))
        if not np.isnan(pn["aic_nb"]):
            fig.add_scatter(x=pn["k"], y=pn["nb_pmf"], name="Negative binomial", mode="lines+markers",
                            line=dict(color="#2e9e6b"))
        fig.update_layout(title="Orders per customer per month", xaxis_title="Orders", yaxis_title="Probability")
        show(fig)
    with r:
        m = st.columns(2)
        m[0].metric("Mean", f"{pn['mean']:.2f}")
        m[1].metric("Variance", f"{pn['var']:.2f}")
        m = st.columns(2)
        m[0].metric("Var / Mean", f"{pn['dispersion']:.2f}")
        m[1].metric("AIC Poisson | NegBin",
                    f"{pn['aic_poisson']:,.0f} | {pn['aic_nb']:,.0f}" if not np.isnan(pn["aic_nb"]) else "n/a")
        st.latex(r"P(N=k)=\frac{e^{-\lambda}\lambda^k}{k!},\quad \mathrm{Var}=\mathrm{Mean}")
    if pn["dispersion"] > 1.2:
        st.info("**So what:** variance is well above the mean (overdispersion), so customers differ a lot in "
                "how often they buy. One average campaign does not fit everyone; target by segment.")
    else:
        st.info("**So what:** variance is close to the mean, so customers buy at similar rates.")

    # ---- 3 gaps -------------------------------------------------------
    st.subheader("3 · Time between purchases: exponential or Weibull?")
    gaps = su.interpurchase_gaps(df)
    e = su.fit_exponential_weibull(gaps)
    hi = np.percentile(gaps, 99)
    xs = np.linspace(0.5, hi, 300)
    l, r = st.columns([3, 2])
    with l:
        fig = go.Figure()
        fig.add_histogram(x=gaps[gaps <= hi], histnorm="probability density", nbinsx=60, name="Observed",
                          marker_color="#9db8ef")
        fig.add_scatter(x=xs, y=stats.expon.pdf(xs, 0, e["mean_gap"]), name="Exponential",
                        line=dict(color="#d9534f", width=3))
        fig.add_scatter(x=xs, y=stats.weibull_min.pdf(xs, e["weibull_shape"], 0, e["weibull_scale"]),
                        name="Weibull", line=dict(color="#2e9e6b", width=3))
        fig.update_layout(title="Days between a customer's purchases", xaxis_title="Days", yaxis_title="Density")
        show(fig)
    with r:
        m = st.columns(2)
        m[0].metric("Mean gap (days)", f"{e['mean_gap']:.1f}")
        m[1].metric("Weibull shape k", f"{e['weibull_shape']:.2f}")
        st.latex(r"f(t)=\lambda e^{-\lambda t},\quad \lambda = 1/\bar t")
        st.latex(r"f(t)=\tfrac{k}{s}\left(\tfrac{t}{s}\right)^{k-1}e^{-(t/s)^k}")
    k = e["weibull_shape"]
    st.info("**So what:** " + ("k below 1 means repeat purchases come soon after an order and then fade, so "
            "follow-up messages should go out early." if k < 0.95 else
            "k above 1 means customers buy on a regular rhythm, so reminders should match that rhythm."
            if k > 1.05 else "k is close to 1, so the timing of the next purchase is close to memoryless."))

    # ---- 4 markov -----------------------------------------------------
    st.subheader("4 · Markov chain: how customers move between states")
    pi = su.steady_state(P)
    mfp = su.mean_first_passage(P)
    l, r = st.columns([3, 2])
    with l:
        show(px.imshow(P, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
                       labels=dict(x="Next month", y="This month", color="P"),
                       title="Transition probability matrix P"))
    with r:
        m = st.columns(2)
        m[0].metric("Active stays Active", f"{P.loc['Active', 'Active']:.0%}")
        m[1].metric("Months until lost (from Active)", f"{mfp.get('Active', np.nan):.1f}")
        st.write("Long-run share of customers: " + ", ".join(f"{k} {v:.0%}" for k, v in pi.items()))
        st.latex(r"P_{ij}=\frac{n_{ij}}{\sum_k n_{ik}},\qquad \pi P=\pi")
        st.latex(r"(I-Q)\,m=\mathbf 1")
    st.info(f"**So what:** each month about {1 - P.loc['Active', 'Active']:.0%} of Active customers slip away, "
            f"and the chain says that without action about {pi['Inactive']:.0%} of customers end up Lost "
            "in the long run. That is the case for acting on At-risk customers early.")

    # ---- 5 hypothesis test -------------------------------------------
    st.subheader("5 · Do Active, Regular and Lost customers really spend differently?")
    grp = {s_: np.log(cust.loc[cust["state"] == s_, "total_spend"].values)
           for s_ in su.STATES if (cust["state"] == s_).sum() >= 3}
    if len(grp) >= 2:
        an = su.anova_suite(grp)
        names = list(grp)
        tt = su.t_test(grp[names[0]], grp[names[-1]])
        m = st.columns(4)
        m[0].metric("ANOVA F", f"{an['F']:.2f}")
        m[1].metric("p-value", f"{an['p']:.2g}")
        m[2].metric("η² (effect size)", f"{an['eta2']:.3f}")
        m[3].metric(f"t-test {names[0]} vs {names[-1]} (p)", f"{tt['p']:.2g}")
        st.markdown("H₀: all group means of log(spend) are equal. " +
                    (f"p = {an['p']:.2g} < α = {alpha} → **reject H₀**." if an["p"] < alpha
                     else f"p = {an['p']:.2g} ≥ α = {alpha} → **fail to reject H₀**."))
        st.latex(r"F=\frac{SS_{between}/(k-1)}{SS_{within}/(N-k)},\qquad \eta^2=\frac{SS_{between}}{SS_{total}}")
        size = "tiny" if an["eta2"] < 0.01 else "small" if an["eta2"] < 0.06 else "medium" if an["eta2"] < 0.14 else "large"
        st.info(f"**So what:** the effect size is {size} (η² = {an['eta2']:.3f}). "
                + ("Customer state clearly relates to how much people spend." if an["p"] < alpha
                   else "No clear spending difference between states was found.")
                + " Log of spend is used because raw spend is lognormal, which breaks the normality "
                  "assumption of ANOVA.")
    else:
        st.warning("Not enough customers in each state for a test.")

# =================================================================== CUSTOMER 360
else:
    st.title("Customer 360")
    ids = list(cust.sort_values("total_spend", ascending=False).index)
    cid = st.selectbox("Customer ID (sorted by spend; type to search)", ids)
    row = cust.loc[cid]
    m = st.columns(5)
    m[0].metric("Segment", row["segment"])
    m[1].metric("Total spend", money(row["total_spend"]))
    m[2].metric("Orders", int(row["n_orders"]))
    m[3].metric("Days since last order", int(row["recency"]))
    m[4].metric("Country", row["country"])

    days = (df[df["customer"] == cid]["date"].dt.normalize().drop_duplicates().sort_values())
    own_gap = days.diff().dt.days.dropna().mean() if len(days) >= 2 else np.nan
    typical = np.median(su.interpurchase_gaps(df)) if len(df) else np.nan
    gap = own_gap if not np.isnan(own_gap) else typical

    onehot = pd.Series([1.0 if s == row["state"] else 0.0 for s in su.STATES], index=su.STATES)
    risk = float(su.project(P, onehot, 3)["Inactive"].iloc[-1])
    level = "Low" if risk < 0.2 else "Medium" if risk < 0.5 else "High"
    st.metric("Risk of becoming a lost customer within 3 months", f"{risk:.0%}", level, delta_color="off")

    st.subheader("Recommended action")
    st.markdown(f"**{row['segment']}:** {su.SEGMENT_ACTIONS[row['segment']]}")
    if not np.isnan(gap):
        who = "This customer usually" if not np.isnan(own_gap) else "Customers typically"
        if row["recency"] > 1.5 * gap and row["segment"] != "Lost":
            st.warning(f"⏰ {who} reorders about every **{gap:.0f} days**, but it has been "
                       f"**{int(row['recency'])} days**. They are overdue: send a reminder now.")
        else:
            st.info(f"{who} reorders about every **{gap:.0f} days**; "
                    f"it has been **{int(row['recency'])} days** since the last order.")
    orders = (df[df["customer"] == cid].groupby("invoice")
              .agg(date=("date", "min"), items=("qty", "sum"), amount=("amount", "sum")).sort_values("date"))
    with st.expander("Order history"):
        table(orders.assign(amount=orders["amount"].round(2)))
