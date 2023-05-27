from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, r_regression


def get_stat_info(df: pd.DataFrame):
    df = df.copy()
    print(df.describe())

    y = df["final_loss"]
    df.drop(columns=["final_loss", "Size", "Generated Text", "total_training_time", "Heads"]
                    + [col for col in df.columns if "Step " in col], inplace=True)

    f_features, p_values = f_regression(df, y)
    mi_features = mutual_info_regression(df, y)
    r_features = r_regression(df, y)

    print(df.columns)
    print(f"F Statistics:\n{f_features}")
    print(f"Mutual Info:\n{mi_features}")
    print(f"Pearson:\n{r_features}")


def display_feature_importance(df: pd.DataFrame, include_training_time: bool):
    df = df.copy()
    clf = RandomForestRegressor()
    if include_training_time:
        y = df[["final_loss", "total_training_time"]]
    else:
        y = df[["final_loss"]]

    df.drop(columns=["final_loss", "total_training_time", "Size", "Generated Text"] +
            [col for col in df.columns if "Step " in col], inplace=True)
    clf.fit(df, y)
    plt.figure(figsize=(12,12))
    plt.bar(df.columns, clf.feature_importances_)
    plt.xticks(rotation=45)
    if include_training_time:
        plt.title("Feature Importance for Loss and Training Time")
    else:
        plt.title("Feature Importance for Loss")
    plt.show()


def plot_hyper_parameter_by(hyper_parameter: str, data: defaultdict, field_prefix: str, label: str):
    for hyper_param_value, values in data.items():
        xs = [row['step'] for row in values]
        ys = [row[f'{field_prefix}_mean'] for row in values]
        y_range_max = [row[f'{field_prefix}_max'] for row in values]
        y_range_min = [row[f'{field_prefix}_min'] for row in values]

        plt.plot(xs, ys, label=f"{hyper_parameter}({hyper_param_value})")
        plt.fill_between(xs, y_range_min, y_range_max, alpha=0.15)
    plt.xscale('symlog')
    # plt.yscale('symlog')
    plt.xlabel("Step")
    plt.ylabel(label)
    plt.legend()
    plt.title(f"{label} by {hyper_parameter} - {max(xs) + 100} Steps")
    # fig.set_ylim(ymin=0)

    plt.show()



def display_hyper_parameter_info(df: pd.DataFrame):
    hyper_parameters = ['LR', 'Heads', 'Embeddings', 'Block Size', 'Batch Size', 'Layers']
    for i, hyper_parameter in enumerate(hyper_parameters):
        print("****")
        fig = plt.figure(i)
        to_drop = [s for s in hyper_parameters if s != hyper_parameter]
        df_copy = df.copy(deep=True)
        df_copy.drop(columns=to_drop, inplace=True)
        print(df_copy.columns)
        df_grouped = df_copy.groupby(by=[hyper_parameter, 'Step']).agg(['mean', 'min', 'max'])

        data = defaultdict(list)  # key is series hyper param value
        for index_name in df_grouped.index:
            hyper_param_value, step = index_name
            # print(index_name)
            train_loss_mean, train_loss_min, train_loss_max, \
                val_loss_mean, val_loss_min, val_loss_max, \
                accum_time_sec_mean, accum_time_sec_min, accum_time_sec_max = df_grouped.loc[index_name].to_list()
            row = {'step': step,
                   'training_loss_mean': train_loss_mean, 'training_loss_min': train_loss_min,
                   'training_loss_max': train_loss_max,
                   'val_loss_mean': val_loss_mean, 'val_loss_min': val_loss_min, 'val_loss_max': val_loss_max,
                   'accum_time_sec_mean': accum_time_sec_mean, 'accum_time_sec_min': accum_time_sec_min,
                   'accum_time_sec_max': accum_time_sec_max
                   }
            data[hyper_param_value].append(row)
            # print(df_grouped.loc[index_name])

        plot_hyper_parameter_by(hyper_parameter, data, "training_loss", "Training Loss")
        plot_hyper_parameter_by(hyper_parameter, data, "val_loss", "Validation Loss")
        plot_hyper_parameter_by(hyper_parameter, data, "accum_time_sec", "Training Time")


if __name__ == '__main__':
    from pathlib import Path
    curdir = Path(".")

    dfs =[]
    #for file_name in curdir.glob("**/output.csv"):
    for file_name in curdir.glob("output.csv"):
        df = pd.read_csv(file_name)
        dfs.append(df)

    df = pd.concat(dfs)
    get_stat_info(df)
    print("-----"*5)

    display_feature_importance(df, include_training_time=True)
    display_feature_importance(df, include_training_time=False)
    print("-----"*5)

    dfs = []
    #for file_name in curdir.glob("**/output_steps.csv"):
    for file_name in curdir.glob("output_steps.csv"):
        df = pd.read_csv(file_name)
        dfs.append(df)

    df = pd.concat(dfs)

    display_hyper_parameter_info(df)

