import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

n = 10

x = np.arange(0.1,0.95,0.2)

df = (
    pd.DataFrame(dict(x=x))
    .assign(
        n1=lambda df: (df.x*n).astype(int),
        data=lambda df: df.n1.map(lambda n1: (n1)*[1]+(n-n1)*[0])
    )
    .explode("data", ignore_index=True)
    .assign(
        x=lambda df: df.x*5,
    )
    .astype(dict(
        data=int
    ))
    .dropna()
)

df.info()
input_cols=["x"]
with pm.Model() as model:
    x = df[input_cols].to_numpy()
    labels = df["data"].to_numpy()

    m = pm.Normal("m", 0, sigma=10, shape=len(input_cols))
    #b=pm.Normal("b", 0, sigma=10)

    score = pm.math.dot(x, m) #+ b
    mu = pm.math.sigmoid(score)
    dist = pm.Bernoulli("dist", p=mu, observed=labels)
    trace = pm.sample(progressbar=True)
#summary = az.summary(trace, var_names=["m", "b"])
summary = az.summary(trace, var_names=["m"])
print(summary.to_markdown())

label_mask = df.data == 0
with pm.Model() as lda:
    mu = pm.Normal("mu", mu=0, sigma=10, shape=2)
    sigma=pm.HalfNormal("sigma",10)

    zero = pm.Normal(
        "zero",
        mu=mu[0],
        sigma=sigma,
        observed=df[df.data==0][input_cols].to_numpy()
    )
    one = pm.Normal(
        "one",
        mu=mu[1],
        sigma=sigma,
        observed=df[df.data==1][input_cols].to_numpy()
    )
    trace_lda=pm.sample()
summary = az.summary(trace_lda, var_names=["mu", "sigma"])
print(summary.to_markdown())