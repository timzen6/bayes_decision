import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

n = 40

x1 = np.arange(0,1.01,0.2)
x2 = np.arange(0,0.11,0.1)

# x1 = [0,0.1,0.9,1]
# x2 = [0,0.05,0.1]
x2 = [0.2]

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
    )
    .assign(
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
    #.sample(frac=1.0, random_state=13, ignore_index=True)
    .assign(
        x1 = lambda df: df.x1.pipe(lambda s: (s-s.mean())/s.std())
    )
)


# input_cols = ["x1", "x2"]
input_cols = ["x1"]
for mixture in (True, False):
    print("mixture" if mixture else "simple")
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

            pi = pm.Beta("pi", 1., 1.)
            if mixture:
                p = pi * 0.5 + (1-pi) * mu
            else:
                p = mu
            pm.Deterministic("p", p)


            # dist = pm.Bernoulli("dist", p=mu, observed=labels)
            dist = pm.Bernoulli("dist", p=p, observed=labels)
            trace = pm.sample(progressbar=False)

        # summary = az.summary(trace, var_names=["m", "b"])
        summary = az.summary(trace, var_names=["m", "pi"])
        print(summary.to_markdown())
        stacked = az.extract(trace)
        df_stacked = pd.DataFrame(dict(
            mu = np.mean(stacked["mu"], axis=1),
            p = np.mean(stacked["p"], axis=1),
        ))
        df_stacked = (
            df_stacked
            .assign(
                x1 = x[:,0],
                label = labels,
                delta = lambda df: np.abs(df.mu-df.p),
            )
        )
        #print(df_stacked.to_markdown())
print("end")
