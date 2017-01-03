import os
import copy
from datetime import datetime

import numpy as np
import pandas as pd

import utils
from gift import Gift
from bag import Bag


class SolutionCandidate:
    def __init__(self, gift_weight_init_method, gift_type_amounts,
                 n_observations_to_evaluate_solution, warm_start_path=None):
        self.bags = self.initialize_bags(
            gift_weight_init_method, gift_type_amounts, warm_start_path)
        self.reward = 0.0
        self.reward_std = 0.0
        self.mean_reject_rate = 0.0
        self.n_observations_to_evaluate_solution = (
            n_observations_to_evaluate_solution)
        self.calculate_reward(n_observations_to_evaluate_solution)

    @staticmethod
    def initialize_bags(gift_weight_init_method, gift_type_amounts,
                        warm_start_path):
        bags = []
        if warm_start_path is None:
            gift_type_amounts_remaining = copy.deepcopy(gift_type_amounts)
            gifts = []
            # Initialize bags
            for _ in range(utils.N_BAGS):
                bags.append(Bag(is_trash_bag=False))
            # Initialize gifts
            for _, row in utils.GIFTS_DF.iterrows():
                gift = Gift(id_label=row['GiftId'],
                            gift_type=row['GiftId'].split('_')[0].title())
                if gift_type_amounts_remaining is not None:
                    gift_type_amounts_remaining[gift.gift_type] -= 1
                if (gift_type_amounts_remaining is None or
                        gift_type_amounts_remaining[gift.gift_type] >= 0):
                    # Add only if gift's type hasn't been added enough already
                    gift.initialize_weight(gift_weight_init_method)
                    gifts.append(gift)
            # Randomize the order of gifts
            np.random.shuffle(gifts)
            # Put gifts in the bags
            all_bags_full = False
            while not all_bags_full:
                n_bags_filled = 0
                # For each bag, add first occurring gift that fits in the bag
                for bag in bags:
                    gift_to_add = None
                    for gift in gifts:
                        if bag.weight + gift.weight <= utils.MAX_BAG_WEIGHT:
                            gift_to_add = gift
                            bag.add_gift(gift_to_add)
                            break
                    if gift_to_add is not None:
                        # Remove added gift from list of gifts
                        gifts.remove(gift_to_add)
                        n_bags_filled += 1
                if n_bags_filled == 0:
                    all_bags_full = True
            if len(gifts) > 0:
                # Add the remaining gifts to a trash bag
                trash_bag = Bag(is_trash_bag=True)
                for gift in gifts:
                    trash_bag.add_gift(gift)
                bags.append(trash_bag)
        else:
            with open(warm_start_path) as f:
                f.readline()  # Throw away first line (header)
                for l in f.readlines():
                    bag = Bag(is_trash_bag=False)
                    for gift_label in l.replace('\n', '').split(' '):
                        gift = Gift(id_label=gift_label,
                                    gift_type=gift_label.split('_')[0].title())
                        gift.initialize_weight(gift_weight_init_method)
                        bag.add_gift(gift)
                    bags.append(bag)
        return bags

    def calculate_reward(self, n_observations_to_evaluate_solution):
        # DataFrames where columns=bags & rows=observations
        simulated_bag_weights = pd.DataFrame()
        bags_rejected = pd.DataFrame()
        for i, bag in enumerate(self.bags):
            if bag.is_trash_bag:
                continue
            bag_weights, bag_rejected = bag.simulate_weight(
                n_observations=n_observations_to_evaluate_solution)
            simulated_bag_weights[str(i)] = bag_weights
            bags_rejected[str(i)] = bag_rejected

        # Mean of simulations
        self.reward = simulated_bag_weights.sum(axis=1).mean()
        # Std of simulations
        self.reward_std = simulated_bag_weights.sum(axis=1).std()
        # Mean of reject rates of simulations
        self.mean_reject_rate = (
            bags_rejected.sum(axis=1) / len(bags_rejected.columns)).mean()

    def mutate(self, mutation_rate=0.001, swap_proba=0.4):
        if isinstance(mutation_rate, float) and 0. < mutation_rate < 1.:
            # Expecting that all gifts are in bags
            n_gifts_mutate = int(np.floor(mutation_rate * utils.N_GIFTS))
        else:
            n_gifts_mutate = int(mutation_rate)
        # Select the bags between which gifts are to be mutated (moved)
        bag_ix_from, bag_ix_to = self.get_mutation_bag_ix(n_gifts_mutate,
                                                          swap_proba)
        # Move gifts between bags
        for bag_from, bag_to in zip(bag_ix_from, bag_ix_to):
            gift = np.random.choice(self.bags[bag_from].gifts)
            self.bags[bag_to].add_gift(gift)
            self.bags[bag_from].remove_gift(gift)

    def get_mutation_bag_ix(self, n_gifts_mutate, swap_proba):
        bag_ix_from = np.random.choice(
            # There must be gift(s) in a bag
            [i for i, bag in enumerate(self.bags) if len(bag.gifts) > 1],
            size=n_gifts_mutate, replace=False).tolist()
        bag_ix_to = []
        for bag_ix in bag_ix_from:
            bag_ix_to.append(np.random.choice(
                # Let's not add gifts to bags where they were taken from
                [i for i in range(len(self.bags)) if i != bag_ix and
                 i not in bag_ix_to]))
        if np.random.uniform(low=0, high=1) < swap_proba:
            # Swap between bags
            bag_ix_from, bag_ix_to = (bag_ix_from + bag_ix_to,
                                      bag_ix_to + bag_ix_from)
        return np.array(bag_ix_from), np.array(bag_ix_to)

    def save_on_hard_drive(self, folder_name):
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Save object
        obj_path = os.path.join(folder_name, '%s--reward=%s--object.npy' % (
            date_time, self.reward))
        variables_save = {'best_individual': self}
        np.save(obj_path, variables_save)
        # Save details.txt & submission.csv
        file_details = open(os.path.join(
            folder_name, '%s--reward=%s--details.txt' % (
                date_time, self.reward)), 'w')
        file_submission = open(os.path.join(
            folder_name, '%s--reward=%s--submission.csv' % (
                date_time, self.reward)), 'w')
        file_submission.write('Gifts\n')
        # Go through gifts in the bags
        for i, bag in enumerate(self.bags, start=1):
            file_details.write('\n\nBag  %-6s Weight %15s\n' % (i, 'Label'))
            file_details.write('----------------------------------\n')
            for j, gift in enumerate(bag.gifts, start=1):
                file_details.write('Gift %-6s %6.2f %15s\n' % (
                    j, gift.weight, gift.id_label))
            file_details.write('----------------------------------\n')
            file_details.write('Total %12.2f\n' % bag.weight)
            if not hasattr(bag, 'is_trash_bag') or (
                        hasattr(bag, 'is_trash_bag') and not bag.is_trash_bag):
                file_submission.write(
                    ' '.join([g.id_label for g in bag.gifts]) + '\n')
        file_details.close()
        file_submission.close()
