import numpy as np
import pandas as pd

n_mail_intervals = [-np.inf, 1, 5, 50, np.inf]

#just a very simplified version
def get_tld_category(tld_ser):
    return tld_ser.mask(~tld_ser.isin(("de", "com")), "unknown")

def prepare_data(df):
    df = (
        df
        [lambda df: df.deltaT==60]
        .assign(
            category = lambda df: get_tld_category(df.Tld),
            nMailLog = lambda df: np.log2(df.nMail)+1,
            nSubdomainLog = lambda df: np.log2(df.nSubdomain)+1,
            isTrusted = lambda df: (df.nTrusted > 0).astype(int),
        )
        .pipe(lambda df: pd.concat([
            df,
            pd.get_dummies(df.category).add_prefix("is_")
        ], axis=1))
        .drop(columns=["category", "is_unknown"])
        .reset_index(drop=True)
    )
    return df


def pre_sampling_stats(df):
    print(
        df.groupby("label")
        .agg(
            dict(
                Domain="count",
                weight="sum",
            )
        )
        .assign(
            rDomain=lambda df: df.Domain.pipe(lambda s: s / s.sum()),
            rWeight=lambda df: df.weight.pipe(lambda s: s / s.sum()),
        )
        .to_markdown()
    )
    return df


def flat_strat_sampling(df, nSample=100, strat_col="mail_strat"):
    cnt = df[strat_col].value_counts()
    nSample_max = len(cnt) * cnt.min()
    assert nSample < nSample_max

    n_strat = int(nSample / len(cnt))

    df_sample = pd.concat(
        [
            df[df[strat_col] == s].sample(n=n_strat, random_state=2)
            for s in df[strat_col].unique()
        ],
        ignore_index=True,
    )

    return df_sample
