import copy
import random
from functools import reduce

import numpy as np
import pandas as pd

RULESET = [lambda x: x[0] < 0.2 and x[2] > 0.7,
           lambda x: 0.5 < x[1] < 0.8,
           lambda x: x[2] < 0.1 and x[1] < 0.1]


def _generate_columns(n_cols):
    return ['x'+str(i) for i in range(n_cols)] + ['y']


def _evaluate_ruleset(ruleset, x):
    return reduce(lambda a, b: a or b, map(lambda rule: rule(x), ruleset))


def _generate_example(n_cols, ruleset, negative_example=False):
    max_trials = 1000
    row = [np.random.random() for _ in range(n_cols)]
    count = 0
    while not _evaluate_ruleset(ruleset, row) if not negative_example else _evaluate_ruleset(ruleset, row) and count < max_trials:
        row = [np.random.random() for _ in range(n_cols)]
        count += 1
    return row


def _generate_free_example(n_cols):
    row = [np.random.random() for _ in range(n_cols)]
    return row


def generate_imbalanced_data(ratio, n_cols, n_rows, ruleset):
    data = []
    for _ in range(n_rows):
        if np.random.random() < ratio:
            row = _generate_example(n_cols, ruleset, negative_example=False)
            row.append(1)
        else:
            row = _generate_example(n_cols, ruleset, negative_example=True)
            row.append(0)
        data.append(row)
    return pd.DataFrame(data, columns=_generate_columns(n_cols))


def generate_uniform_data(n_cols, n_rows, ruleset):
    data = []
    for _ in range(n_rows):
        row = _generate_free_example(n_cols)
        label = _evaluate_ruleset(ruleset, row)
        row.append(label)
        data.append(row)
    return pd.DataFrame(data, columns=_generate_columns(n_cols))


def _choose_other(xs, x):
    xs_copy = copy.copy(xs)
    xs_copy.remove(x)
    return random.choice(xs_copy)


def make_noisy(df, label_col, error_rate):
    # Note that this changes the pos ratio if pos ratio is not 0.5
    df_copy = copy.copy(df)
    labels = list(df_copy[label_col].unique())
    df_copy[label_col] = df_copy[label_col].apply(lambda x: x if np.random.random() > error_rate else _choose_other(labels, x))
    return df_copy


if __name__ == "__main__":
    this_df = generate_imbalanced_data(0.2, 4, 100, RULESET)
    print(this_df['y'].value_counts())
    print(this_df)
