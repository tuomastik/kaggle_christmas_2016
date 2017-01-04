import os

import numpy as np
import pandas as pd


GIFTS_DF = pd.read_csv(os.path.join('data', 'gifts.csv'))
GIFTS_DF['gift_type'] = GIFTS_DF['GiftId'].apply(  # Add new column gift_type
    lambda x: x.split('_')[0].title())


N_BAGS = 1000
N_GIFTS = GIFTS_DF.shape[0]  # 7166
MIN_GIFTS_IN_BAG = 3
MAX_BAG_WEIGHT = 50


GIFT_WEIGHT_DISTRIBUTIONS = {
    'horse': lambda: max(0, np.random.normal(5, 2, 1)[0]),
    'ball': lambda: max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
    'bike': lambda: max(0, np.random.normal(20, 10, 1)[0]),
    'train': lambda: max(0, np.random.normal(10, 5, 1)[0]),
    'coal': lambda: 47 * np.random.beta(0.5, 0.5, 1)[0],
    'book': lambda: np.random.chisquare(2, 1)[0],
    'doll': lambda: np.random.gamma(5, 1, 1)[0],
    'blocks': lambda: np.random.triangular(5, 10, 20, 1)[0],
    'gloves': lambda: (3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3
                       else np.random.rand(1)[0])}


def simulate_gift_weights(n_observations_per_gift=1000):
    # Get unique gift types
    gift_types = GIFTS_DF['GiftId'].apply(lambda x: x.split('_')[0]).unique()
    # Draw observations of weights from each gift type weight distributions
    simulated_data = pd.DataFrame()
    for gift_type in gift_types:
        simulated_data[gift_type.title()] = [
            GIFT_WEIGHT_DISTRIBUTIONS[gift_type]() for _ in
            range(n_observations_per_gift)]
    return simulated_data


SIMULATED_GIFTS = simulate_gift_weights(n_observations_per_gift=100000)
EXPECTED_GIFT_WEIGHTS = {}


def set_expected_gift_weights(approach, n_observations_per_gift=None):
    # Examples using n_observations_per_gift=1000000:
    # Mean: {'Horse': 5.0099736513214941, 'Train': 10.020979390391856,
    #        'Ball': 1.9998198370591844, 'Gloves': 1.4046643389013214,
    #        'Doll': 5.0065261219194594, 'Blocks': 11.676912447098612,
    #        'Coal': 23.463422545117151, 'Bike': 20.090732813933034,
    #        'Book': 2.0037273036259213}
    # Median: {'Horse': 5.0019744540973088, 'Train': 9.9882838205225486,
    #          'Ball': 1.9995971119135794, 'Gloves': 0.71743160618285484,
    #          'Doll': 4.6863390890278813, 'Blocks': 11.347290397556351,
    #          'Coal': 23.402859280337825, 'Bike': 19.960136626060969,
    #          'Book': 1.3873168037639485}
    global EXPECTED_GIFT_WEIGHTS
    if approach == 'numerical_mean' or approach == 'numerical_median':
        simulated_data = simulate_gift_weights(n_observations_per_gift)
        if approach == 'numerical_mean':
            EXPECTED_GIFT_WEIGHTS = simulated_data.mean(axis=0).to_dict()
        else:
            EXPECTED_GIFT_WEIGHTS = simulated_data.median(axis=0).to_dict()
    elif approach == 'analytical':
        # Determined from parameters of distribution declaration
        EXPECTED_GIFT_WEIGHTS = {
            'Horse': 5.0,
            'Ball': 2.0,
            'Bike': 20.0,
            'Train': 10.0,
            'Coal': 47 * (0.5 / (0.5 + 0.5)),  # = 23.5
            'Book': 2.0,
            'Doll': 5.0,
            'Blocks': (5 + 10 + 20) / 3.0,  # = 11.67
            'Gloves': 0.5 * 0.7 + 0.3 * 3.5}  # = 1.4
    else:
        raise(Exception("Unknown 'approach' parameter value: %s" % approach))


def round_down_to_even(f):
    return int(np.floor(f / 2.) * 2)
