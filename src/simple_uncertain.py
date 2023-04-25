import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

n = 20

x1 = np.arange(0,1.01,0.2)
x2 = np.arange(0,0.11,0.1)

# x1 = [0,0.1,0.9,1]
x2 = [0,0.05,0.1]
#x2 = [0]

df_raw = pd.DataFrame([
    dict(
        x1=a,
        x2=b,
    )
    for b in x2
    for a in x1
])

df_raw = (
    df_raw
    .assign(
        n0 = lambda df: df.x1.map(lambda x: int(x*n)),
        n_err = lambda df: df.apply(lambda row:
            row.n0 + (row.x2*n) if row.x1 < 0.5 else
            row.n0 - (row.x2*n), axis=1 
        ).astype(int),
        label = lambda df: df.n0.map(lambda i: i*[1]+((n-i)*[0])),
        label_err = lambda df: df.n_err.map(lambda i: i*[1]+((n-i)*[0])),
    )
)

df = (
    df_raw
    .explode(["label", "label_err"], ignore_index=True)
    .astype(dict(
        label = int,
        label_err = int,
    ))
    .sample(frac=1.0, random_state=13, ignore_index=True)
    .assign(
        x1 = lambda df: df.x1.pipe(lambda s: (s-s.mean())/s.std())
    )
)


input_cols = ["x1", "x2"]
# input_cols = ["x1"]
for l in ("label", "label_err"):
    with pm.Model() as model:
        x = df[input_cols].to_numpy()
        labels = df[l].to_numpy()

        m = pm.Normal("m", 0, sigma=10, shape=len(input_cols))
        # b=pm.Normal("b", 0, sigma=10)

        # score = pm.math.dot(x, m) + b
        score = pm.math.dot(x, m)
        mu = pm.math.sigmoid(score)

        pm.Deterministic("mu", mu)
        dist = pm.Bernoulli("dist", p=mu, observed=labels)
        trace = pm.sample()

    # summary = az.summary(trace, var_names=["m", "b"])
    summary = az.summary(trace, var_names=["m"])
    print(summary.to_markdown())
print("end")
