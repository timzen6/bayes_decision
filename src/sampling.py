import numpy as np
import pandas as pd

n_mail_intervals = [-np.inf, 1, 5, 50, np.inf]


def prepare_data(df):
    df = df.assign(
        mail_strat=lambda df: pd.cut(
            df.nMail, n_mail_intervals, labels=range(len(n_mail_intervals) - 1)
        ),
        # weight = lambda df: np.log10(df.nMail)+1,
        weight=lambda df: df.nMail,
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
    print(
        df.assign(
            w_mal=lambda df: df.label * df.weight,
        )
        .groupby("mail_strat")
        .agg(
            dict(
                Domain="count",
                weight="sum",
                w_mal="sum",
                label="sum",
            )
        )
        .assign(
            rStrat=lambda df: df.Domain / df.Domain.sum(),
            wStrat=lambda df: df.weight / df.weight.sum(),
            rlabel=lambda df: df.label / df.Domain,
            wlabel=lambda df: df.w_mal / df.weight,
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
