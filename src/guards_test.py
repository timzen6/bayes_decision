import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss

from sampling import prepare_data, pre_sampling_stats, flat_strat_sampling

# df = pd.read_csv("../contracted_test_data/new_domains/2022-10-31-28/train.csv")
# df_compact = pd.read_csv("../contracted_test_data/new_domains/2022-10-31-28/train_compact.csv")

df = pd.read_csv("data/contracted_test_data/new_domains/2022-10-31-28/train.csv")
print(df.head().to_markdown())

df = (
    df
    .pipe(prepare_data)
    .pipe(pre_sampling_stats)
)
df_sample = (
    flat_strat_sampling(df, 200)
    .pipe(pre_sampling_stats)
)


df_compact = (
    df_compact
    # .sample(n=10000, weights="weight", random_state=2)
    .sample(n=10000, random_state=2)
)

df_compact = pd.concat(
    [df_compact, pd.get_dummies(df_compact.category).add_prefix("is_")], axis=1
).drop(columns=["is_unknown", "category"])

all_input_cols = [c for c in df_compact.columns if c not in ("label", "weight")]
print(df_compact.columns)

input_cols = ["nMail", "nSubdomain", "is_com", "is_de"]
# input_cols = [c for c in all_input_cols if (c not in ["nSubdomain5"] and not c.startswith("keyword"))]

with pm.Model() as model:
    x = df_compact[input_cols].to_numpy()
    # w = df_compact["weight"].to_numpy()
    w = np.log10(df_compact["weight"].to_numpy())
    labels = df_compact["label"].to_numpy()

    m = pm.Normal("m", 0, sigma=10, shape=len(input_cols))
    b = pm.Normal("b", 0, sigma=10)

    score = pm.math.dot(x, m) + b
    mu = pm.math.sigmoid(score)

    pm.Deterministic("mu", mu)

    # test = pm.Bernoulli("test", p=mu, observed=labels)

    logp = w * pm.logp(pm.Bernoulli.dist(p=mu), labels)
    pm.Deterministic("logp", logp)
    error = pm.Potential("error", logp)
    trace = pm.sample()

summary = az.summary(trace, var_names=["m", "b"]).assign(
    input_names=input_cols + ["intercept"]
)
summary = summary[["input_names"] + [c for c in summary.columns if c != "input_names"]]
print(summary.to_markdown())

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
    df_compact.assign(
        dummy="unkown",
        cat=lambda df: df.dummy.mask(df.is_de > 0, "de").mask(df.is_com > 0, "com"),
        mu=np.mean(stacked.mu, axis=1),
        mu_err=np.std(stacked.mu, axis=1),
    )
    .sample(n=min(500, len(df_compact)), random_state=42)
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
