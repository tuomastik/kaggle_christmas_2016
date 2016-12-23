import os

import numpy as np
import pandas as pd

import utils
import viz


if __name__ == '__main__':
    seed = 1122345
    np.random.seed(seed)
    gifts_df = pd.read_csv(os.path.join('data', 'gifts.csv'))
    n_observations_per_gift = 1000
    simulated_weights = utils.simulate_gift_weights(
        df=gifts_df, n_observations_per_gift=n_observations_per_gift)
    viz.visualize_gift_type_counts(df=gifts_df)
    viz.visualize_gift_type_weight_distributions(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
    viz.visualize_gift_type_weight_box_plots(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
