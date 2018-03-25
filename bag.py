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
        if self.is_trash_bag or len(self.gifts) < 3:
            bag_rejected = np.ones(shape=(1, n_observations))
            bag_weights = np.zeros(shape=(1, n_observations))
        else:
            bag_weights = utils.SIMULATED_GIFTS[
                [g.gift_type for g in self.gifts]].apply(
                np.random.permutation, axis=0).sum(axis=1)
            bag_rejected = []
            for i, bag_weight in enumerate(bag_weights):
                if bag_weight <= utils.MAX_BAG_WEIGHT:
                    bag_rejected.append(0)
                else:
                    bag_weights[i] = 0.0
                    bag_rejected.append(1)
        return bag_weights.tolist(), bag_rejected
