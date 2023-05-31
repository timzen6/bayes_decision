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

df_clean = df[df.label==0].reset_index(drop=True)
df_mal = df[df.label==1].reset_index(drop=True)

az.plot_kde(
    values = df_clean.nMail,
    label="clean"
)
plt.show()
plt.clf()
az.plot_kde(
    values = df_mal.nMail,
    label="mal"
)
plt.show()
#1d test
#nd test