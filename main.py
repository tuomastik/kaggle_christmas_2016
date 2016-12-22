import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm


LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10


GIFT_WEIGHT_DISTRIBUTIONS = {
    'horse': lambda: max(0, np.random.normal(5, 2, 1)[0]),
    'ball': lambda: max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
    'bike': lambda: max(0, np.random.normal(20, 10, 1)[0]),
    'train': lambda: max(0, np.random.normal(10, 5, 1)[0]),
    'coal': lambda: 47 * np.random.beta(0.5, 0.5, 1)[0],
    'book': lambda: np.random.chisquare(2, 1)[0],
    'doll': lambda: np.random.gamma(5, 1, 1)[0],
    'blocks': lambda: np.random.triangular(5, 10, 20, 1)[0],
    'gloves': lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3
        else np.random.rand(1)[0]}


def simulate_gift_weights(df, n_observations_per_gift=1000):
    # Get unique gift types
    gift_types = df['GiftId'].apply(lambda x: x.split('_')[0]).unique()
    # Draw observations of weights from each gift type weight distributions
    simulated_data = pd.DataFrame()
    for gift_type in gift_types:
        simulated_data[gift_type.title()] = [
            GIFT_WEIGHT_DISTRIBUTIONS[gift_type]() for _ in
            range(n_observations_per_gift)]
    return simulated_data


def visualize_gift_type_counts(df):
    # Get unique gift types and their counts
    gift_types = df['GiftId'].apply(lambda x: x.split('_')[0]).value_counts()
    # Capitalize gift types
    gift_types.index = [gift.title() for gift in gift_types.index]
    # Generate colors for bars
    colormap = cm.get_cmap('Spectral')
    norm = colors.Normalize(vmax=gift_types.max(), vmin=gift_types.min())
    bar_colors = [colormap(norm(val)) for val in gift_types]
    # Visualize
    ax = gift_types.plot.barh(figsize=(10, 7), fontsize=TICK_FONTSIZE,
                              color=bar_colors, alpha=0.8,
                              title="Counts of gift types in the "
                                    "provided data (gifts.csv)")
    ax.set_ylabel("Gift type", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Count", fontsize=LABEL_FONTSIZE)
    plt.tight_layout()
    plt.savefig('gift_type_counts.png')


def visualize_gift_type_weight_distributions(df, n_observations_per_gift):
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("KDE and histogram of %s simulated weights for each "
                 "gift type" % n_observations_per_gift,
                 fontsize=LABEL_FONTSIZE, y=0.93)
    for i, gift_type in enumerate(df.columns, start=1):
        # Plot distribution of weights for each gift type
        ax = fig.add_subplot(3, 3, i)
        sns.distplot(df[gift_type], ax=ax, kde_kws={
            'clip': (np.min(df[gift_type]), np.max(df[gift_type])),
            'color': '#002F2F'}, hist_kws={'color': '#046380'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(6)
    plt.savefig('gift_type_weight_distributions.png')


def visualize_gift_type_weight_box_plots(df, n_observations_per_gift):
    ax = plt.figure(figsize=(10, 7)).add_subplot(111)
    sns.boxplot(data=df, ax=ax, linewidth=0.7, fliersize=3,
                palette=sns.color_palette(
                    palette="Spectral", n_colors=len(df.columns)))
    mi, ma, std = df.values.min(), df.values.max(), df.values.std()
    height_extend = 0.3
    ax.set_ylim((mi - height_extend * std, ma + height_extend * std))
    ax.set_ylabel("Weight (pound)", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Gift type", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.set_title("Box plots of %s simulated weights for each gift type." %
                 n_observations_per_gift)
    plt.tight_layout()
    plt.savefig('gift_type_weight_box_plots.png')


if __name__ == '__main__':
    gifts_df = pd.read_csv(os.path.join('data', 'gifts.csv'))
    n_observations_per_gift = 1000
    simulated_weights = simulate_gift_weights(
        df=gifts_df, n_observations_per_gift=n_observations_per_gift)
    visualize_gift_type_counts(df=gifts_df)
    visualize_gift_type_weight_distributions(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
    visualize_gift_type_weight_box_plots(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
