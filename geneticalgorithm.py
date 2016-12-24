import numpy as np

import utils


GIFTS_PER_BAG_INITIALLY = np.diff(np.linspace(
    start=0, stop=utils.N_GIFTS, num=utils.N_BAGS + 1, dtype=int))


class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.individuals = [SolutionCandidate() for _ in
                            range(population_size)]


class SolutionCandidate:
    def __init__(self):
        print("Creating solution...")
        self.bags = self.initialize_bags()
        self.reward = 0
        self.calculate_reward()
        print("  - Reward: %s" % self.reward)

    @staticmethod
    def initialize_bags():
        # print("Initializing bags...")
        bags = []
        random_row_indices = np.random.permutation(utils.N_GIFTS)
        for n_gifts in GIFTS_PER_BAG_INITIALLY:
            bag = Bag()
            extracted_gifts = utils.GIFTS_DF.iloc[random_row_indices[:n_gifts]]
            random_row_indices = random_row_indices[n_gifts:]
            for row_ix, gift in extracted_gifts.iterrows():
                bag.add_gift(
                    weight=utils.EXPECTED_GIFT_WEIGHTS[gift['gift_type']],
                    id_label=gift['GiftId'],
                    running_number=row_ix)
            bags.append(bag)
        return bags

    def calculate_reward(self):
        self.reward = 0
        for bag in self.bags:
            if bag.weight <= utils.MAX_BAG_WEIGHT:
                self.reward += bag.weight
            # else:
            #     print("  - Too heavy bag!")


class Bag:
    def __init__(self):
        self.gifts = []
        self.weight = 0.0

    def add_gift(self, weight, id_label, running_number):
        self.gifts.append(Gift(weight, id_label, running_number))
        self.weight += weight


class Gift:
    def __init__(self, weight, id_label, running_nr):
        self.weight = weight
        self.id_label = id_label
        self.running_nr = running_nr
