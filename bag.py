import numpy as np

import utils


class Bag:
    def __init__(self, is_trash_bag=False):
        self.gifts = []
        self.weight = 0.0
        self.is_trash_bag = is_trash_bag

    def add_gift(self, gift):
        self.gifts.append(gift)
        self.weight += gift.weight

    def remove_gift(self, gift):
        self.gifts.remove(gift)
        self.weight -= gift.weight

    def simulate_weight(self, n_observations=1000):
        if self.is_trash_bag:
            bag_rejected = np.ones(shape=(1, n_observations))
            bag_weights = np.zeros(shape=(1, n_observations))
        else:
            # Simulate bag weight n_observations times
            bag_weights = np.zeros(n_observations)
            row_indices = np.random.permutation(utils.SIMULATED_GIFTS.shape[0])
            for i, gift in enumerate(self.gifts):
                bag_weights += utils.SIMULATED_GIFTS[gift.gift_type].iloc[
                    row_indices[i*n_observations:(i+1)*n_observations]]
            bag_rejected = []
            for i, bag_weight in enumerate(bag_weights):
                if len(self.gifts) >= 3 and bag_weight <= utils.MAX_BAG_WEIGHT:
                    bag_rejected.append(0)
                else:
                    bag_weights[i] = 0.0
                    bag_rejected.append(1)
        return bag_weights.tolist(), bag_rejected
