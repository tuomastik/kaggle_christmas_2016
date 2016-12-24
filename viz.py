import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt

import utils


FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
KDE_NOTICE = ("Do note that weights cannot really get negative values, "
              "it's just the prolonged estimates of the distributions.")


def visualize_gift_type_counts():
    # Get unique gift types and their counts
    gift_types = utils.GIFTS_DF['GiftId'].apply(
        lambda x: x.split('_')[0]).value_counts()
    # Capitalize gift types
    gift_types.index = [gift.title() for gift in gift_types.index]
    # Generate colors for bars
    colormap = cm.get_cmap('Spectral')
    norm = colors.Normalize(vmax=gift_types.max(), vmin=gift_types.min())
    bar_colors = [colormap(norm(val)) for val in gift_types]
    # Visualize
    ax = gift_types.plot.barh(figsize=(10, 7), fontsize=FONTSIZE_TICK,
                              color=bar_colors, alpha=0.8,
                              title="Counts of gift types in the "
                                    "provided data (gifts.csv)")
    ax.set_ylabel("Gift type", fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Count", fontsize=FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig('gift_type_counts.png')


def visualize_gift_type_weight_distributions(df, n_observations_per_gift):
    # Separately
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("KDE and histogram of %s simulated weights separately for "
                 "each gift type.\n%s" % (n_observations_per_gift, KDE_NOTICE),
                 fontsize=FONTSIZE_LABEL, y=0.96)
    for i, gift_type in enumerate(df.columns, start=1):
        # Plot distribution of weights for each gift type
        ax = fig.add_subplot(3, 3, i)
        sns.distplot(df[gift_type], ax=ax, kde_kws={'color': '#002F2F'},
                     hist_kws={'color': '#046380'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(6)
    plt.savefig('gift_type_weight_distributions_separately.png')
    # Stacked
    # -------------------------------------------------------------------------
    ax = plt.figure(figsize=(10, 7)).add_subplot(111)
    min_x, max_x, max_y = 0, 0, 0
    for gift_type, color in zip(df.columns, sns.color_palette(
            palette="Paired", n_colors=len(df.columns))):
        ax_kde = sns.kdeplot(df[gift_type], ax=ax, linewidth=3, color=color,
                             alpha=0.8)
        min_x_current, max_x_current = ax_kde.get_xlim()
        min_y_current, max_y_current = ax_kde.get_ylim()
        if max_y_current > max_y:
            max_y = max_y_current
        if min_x_current < min_x:
            min_x = min_x_current
        if max_x_current > max_x:
            max_x = max_x_current
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((-0.005, max_y+0.01))
    ax.set_title("Stacked KDEs of %s simulated weights for each gift type.\n"
                 "%s" % (n_observations_per_gift, KDE_NOTICE),
                 fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Weight (pound)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Density", fontsize=FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig('gift_type_weight_distributions_stacked.png')


def visualize_gift_type_weight_box_plots(df, n_observations_per_gift):
    ax = plt.figure(figsize=(10, 7)).add_subplot(111)
    sns.boxplot(data=df, ax=ax, linewidth=0.7, fliersize=3,
                palette=sns.color_palette(
                    palette="Spectral", n_colors=len(df.columns)))
    mi, ma, std = df.values.min(), df.values.max(), df.values.std()
    height_extend = 0.3
    ax.set_ylim((mi - height_extend * std, ma + height_extend * std))
    ax.set_ylabel("Weight (pound)", fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Gift type", fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
    ax.set_title("Box plots of %s simulated weights for each gift type" %
                 n_observations_per_gift)
    plt.tight_layout()
    plt.savefig('gift_type_weight_box_plots.png')


if __name__ == '__main__':
    seed = 1122345
    np.random.seed(seed)
    n_observations_per_gift = 1000
    simulated_weights = utils.simulate_gift_weights(n_observations_per_gift)
    visualize_gift_type_counts()
    visualize_gift_type_weight_distributions(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
    visualize_gift_type_weight_box_plots(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
