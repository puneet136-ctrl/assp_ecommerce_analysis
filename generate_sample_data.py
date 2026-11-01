"""
generate_sample_data.py - builds a realistic synthetic transaction file with the
same columns as the UCI "Online Retail" dataset, so the dashboard runs even
before you download the real data.

Run:  python generate_sample_data.py     ->  sample_transactions.csv
"""
import numpy as np
import pandas as pd

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium"]
COUNTRY_P = [0.78, 0.07, 0.05, 0.03, 0.03, 0.02, 0.02]


def make_sample(n_customers: int = 1500, seed: int = 42,
                start: str = "2010-12-01", end: str = "2011-12-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_days = (end - start).days

    n_products = 40
    stock = [f"{85000 + i}" for i in range(n_products)]
    desc = [f"PRODUCT ITEM {i:02d}" for i in range(n_products)]
    base_price = np.round(rng.lognormal(1.0, 0.7, n_products), 2).clip(0.29, 40)

    hours = np.arange(8, 19)
    hour_w = np.array([2, 5, 8, 9, 10, 12, 10, 8, 6, 4, 2], float)
    hour_w /= hour_w.sum()

    rows, invoice_no = [], 536365
    for cid in range(12346, 12346 + n_customers):
        join_day = rng.beta(1.0, 2.2) * total_days
        rate = rng.gamma(0.8, 2.0) / 30.0                       # orders/day (gamma mixture -> overdispersion)
        lifetime = rng.exponential(130) + 25 if rng.random() < 0.85 else rng.exponential(500)
        spend_mult = rng.lognormal(0, 0.55) * (8 if rng.random() < 0.01 else 1)
        country = rng.choice(COUNTRIES, p=COUNTRY_P)
        t = join_day
        while True:
            if t > min(join_day + lifetime, total_days):
                break
            day = start + pd.Timedelta(days=float(t))
            if day.weekday() == 5:                              # no Saturday trading (as in real data)
                day += pd.Timedelta(days=1)
            ts = day.normalize() + pd.Timedelta(hours=int(rng.choice(hours, p=hour_w)),
                                                minutes=int(rng.integers(0, 60)))
            n_lines = 1 + rng.poisson(3.5)
            for _ in range(n_lines):
                k = rng.integers(0, n_products)
                qty = max(1, int(rng.geometric(0.3) * spend_mult))
                price = round(float(base_price[k] * rng.lognormal(0, 0.08)), 2)
                rows.append((str(invoice_no), stock[k], desc[k], qty, ts, price, cid, country))
                if rng.random() < 0.01:                         # occasional return/cancellation
                    rows.append(("C" + str(invoice_no), stock[k], desc[k], -qty,
                                 ts + pd.Timedelta(days=2), price, cid, country))
            invoice_no += 1
            t += rng.exponential(1.0 / max(rate, 1e-4))

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    return df.sort_values("InvoiceDate").reset_index(drop=True)


if __name__ == "__main__":
    data = make_sample()
    data.to_csv("sample_transactions.csv", index=False)
    print(f"Wrote sample_transactions.csv with {len(data):,} rows, "
          f"{data['CustomerID'].nunique():,} customers")
