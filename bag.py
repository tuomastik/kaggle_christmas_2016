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
        bag_weights, bag_rejected = [], []
        for _ in range(n_observations):
            gift_weigths = np.array([
                utils.GIFT_WEIGHT_DISTRIBUTIONS[gift.gift_type.lower()]()
                for gift in self.gifts])
            simulated_bag_weight = gift_weigths.sum()
            if (not self.is_trash_bag and len(self.gifts) >= 3 and
                    simulated_bag_weight <= utils.MAX_BAG_WEIGHT):
                bag_weights.append(simulated_bag_weight)
                bag_rejected.append(0)
            elif not self.is_trash_bag:
                bag_weights.append(0)
                bag_rejected.append(1)
        return bag_weights, bag_rejected
