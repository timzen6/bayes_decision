import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss

from sampling import flat_strat_sampling, pre_sampling_stats, prepare_data


print("Training")
df = pd.read_parquet("data/train.parquet")
df = df.pipe(prepare_data).pipe(pre_sampling_stats)

print("Test")
df_test = (
    pd.read_parquet("data/test.parquet")
    .pipe(prepare_data)
    .pipe(pre_sampling_stats)
)

all_input_cols = [c for c in df.columns if c not in ("label", "weight")]
print(df.columns)

input_cols = ["nMailLog", "nSubdomainLog", "is_com", "is_de", "isTrusted"]
# input_cols = [c for c in all_input_cols if (c not in ["nSubdomain5"] and not c.startswith("keyword"))]

# TODO: basic normalizing of inputs 
x_input =  x = df[input_cols].to_numpy()
# w = df["weight"].to_numpy()
w = ((len(df)/df.weight.sum())*df["weight"]).to_numpy()
# w = np.log10(df["weight"].to_numpy()) + 1
labels = df["label"].to_numpy()

with pm.Model() as model:

    x = pm.MutableData("x", x_input)
    m = pm.Normal("m", 0, sigma=10, shape=len(input_cols))
    b = pm.Normal("b", 0, sigma=10)

    score = pm.math.dot(x, m) + b
    mu = pm.math.sigmoid(score)

    pm.Deterministic("mu", mu)

    # test = pm.Bernoulli("test", p=mu, observed=labels)

    logp = w * pm.logp(pm.Bernoulli.dist(p=mu), labels)
    pm.Deterministic("logp", logp)
    error = pm.Potential("error", logp)
    trace = pm.sample(500)

summary = az.summary(trace, var_names=["m", "b"]).assign(
    input_names=input_cols + ["intercept"]
)
summary = summary[["input_names"] + [c for c in summary.columns if c != "input_names"]]
print(summary.to_markdown())

# Performance calculations with any data on fitted model
with model:
    pm.set_data(dict(
        x=df_test[input_cols]
    ))
    ppc = pm.sample_posterior_predictive(trace, var_names=["mu"])

ppc_mu = ppc.posterior_predictive.mu
df_prediction = pd.DataFrame(dict(
    mu_avg = np.mean(ppc_mu, axis=(0,1)),
    mu_std = np.std(ppc_mu, axis=(0,1)),
    label = df_test.label,
    weight=df_test.weight
))
print(df_prediction.head(20).to_markdown())
print(accuracy_score(
    df_prediction.label,
    (df_prediction.mu_avg>0.5).astype(int),
    sample_weight=df_prediction.weight,
))
exit()

az.plot_trace(trace, var_names=["m", "b"])
plt.savefig("figures/guards_trace.png")

stacked = az.extract(trace)
result = pd.DataFrame(
    dict(
        logp_avg=np.mean(stacked["logp"], axis=1),
        logp_sd=np.std(stacked["logp"], axis=1),
        weight=w,
        label=labels,
        mu_avg=np.mean(stacked["mu"], axis=1),
        mu_5=np.percentile(stacked["mu"], 5, axis=1),
        mu_95=np.percentile(stacked["mu"], 95, axis=1),
        mu_sd=np.std(stacked["mu"], axis=1),
    )
).assign(mu_delta=lambda df: df.mu_95 - df.mu_5)
result.to_csv("data/result_tmp.csv", index=False)

print(
    result.sort_values("mu_delta", ascending=False, ignore_index=True)
    .head(25)
    .to_markdown()
)
print(
    result.groupby("label")
    .agg(
        dict(
            weight="sum",
            mu_avg="count",
            mu_sd="mean",
            mu_delta="mean",
        )
    )
    .to_markdown()
)


df_score = (
    result[["logp_avg", "weight"]]
    .sum()
    .to_frame()
    .transpose()
    .assign(
        score=lambda df: np.exp(df.logp_avg / df.weight),
        log_loss=log_loss(result.label, result.mu_avg, sample_weight=result.weight),
        acc_score=accuracy_score(
            result.label, (result.mu_avg > 0.5).astype(int), sample_weight=result.weight
        ),
    )
)
print(df_score.to_markdown())


fig = px.scatter(
    df.assign(
        dummy="unkown",
        cat=lambda df: df.dummy.mask(df.is_de > 0, "de").mask(df.is_com > 0, "com"),
        mu=np.mean(stacked.mu, axis=1),
        mu_err=np.std(stacked.mu, axis=1),
    )
    .sample(n=min(500, len(df)), random_state=42)
    .astype(dict(label="str")),
    x="nMail",
    y="mu",
    error_y="mu_err",
    size="weight",
    size_max=10,
    # color = "cat",
    color="label",
)
fig.write_image("figures/guards_prediction.png")
