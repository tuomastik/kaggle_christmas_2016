import os
import copy
from datetime import datetime

import numpy as np

import utils


SAVED_SOLUTIONS_FOLDER = 'ga_solutions'
SIMULATED_GIFTS = utils.simulate_gift_weights(n_observations_per_gift=1000)


class SelectionMethod:
    def __init__(self):
        pass
    truncation = 'truncation'
    roulette_wheel = 'roulette_wheel'


class GiftWeightInitMethod:
    def __init__(self):
        pass
    expected_mean = 'numerical_mean'
    expected_median = 'numerical_median'
    expected_analytical = 'analytical'
    sample_from_distr = 'sample_from_distribution'


class BasinHopping:

    def __init__(self, population_size=50,
                 gift_weight_init_method=GiftWeightInitMethod.expected_mean,
                 gift_type_amounts=None):
        self.results_folder_name = self.save_init_settings_on_hard_drive(
            population_size, gift_weight_init_method, gift_type_amounts)
        self.calculate_expected_weights_if_needed(gift_weight_init_method)
        self.individuals = []
        for i in range(1, population_size + 1):
            print("\rInitializing population - solution candidate %s / %s" % (
                  i, population_size), end='', flush=True)  # Print same line
            self.individuals.append(SolutionCandidate(
                gift_weight_init_method, gift_type_amounts))
        self.individuals = np.array(self.individuals)
        self.best_individual = None

    def find_optimal_params(self):
        print("\nStarting hopping:...")

        # Pre calculate things.
        # Round the number of parents to be selected to an even number so that
        # we can execute bio-inspired operators in a pair wise manner.
        n_parents = utils.round_down_to_even(for_reproduction *
                                             len(self.individuals))
        if n_parents == 0:
            raise(Exception("Zero number of parents. Check parameters "
                            "population_size and for_reproduction."))
        children_per_parent = int(1.0 / for_reproduction)

        # Save settings
        self.save_train_settings_on_hard_drive(
            n_parents, children_per_parent, n_generations, mutation_rate,
            selection_method)

        # Evolution
        for generation in range(1, n_generations + 1):

            parents = self.select_parents(selection_method, n_parents)

            print("\nGeneration %s" % generation)
            print("Rewards of the new parents\n--------------------------")
            for p in parents:
                print("Reward: %.3f - Std of expected weights of bags: %.3f" %
                      (p.reward, p.bags_expected_weights_std))

            self.individuals = self.breed_mutation(
                parents=parents, children_per_parent=children_per_parent,
                mutation_rate=mutation_rate)
            self.calculate_rewards()

            self.find_best_individual()
            if generation % 20 == 0:
                self.best_individual.save_on_hard_drive(
                    folder_name=self.results_folder_name)
        print("Training complete!")

    def parent_selection_roulette_wheel(self, n_parents):
        rewards = np.array([i.reward for i in self.individuals])
        total_reward = rewards.sum()
        relative_rewards = rewards / float(total_reward)
        probability_intervals = relative_rewards.cumsum()
        parents = []
        for _ in range(n_parents):
            random_probability = np.random.uniform(low=0, high=1, size=1)[0]
            for i, p in zip(self.individuals, probability_intervals):
                if random_probability <= p:
                    parents.append(i)
                    break
        return parents

    def parent_selection_truncation(self, n_parents):
        rewards = [i.reward for i in self.individuals]
        # Find indices of solutions with best rewards
        best_ix = np.argpartition(rewards, -n_parents)[-n_parents:]
        parents = self.individuals[best_ix]
        return parents

    def select_parents(self, selection_method, n_parents):
        if selection_method == SelectionMethod.truncation:
            return self.parent_selection_truncation(n_parents)
        elif selection_method == SelectionMethod.roulette_wheel:
            return self.parent_selection_roulette_wheel(n_parents)
        else:
            raise(Exception("Unknown selection_method"))

    @staticmethod
    def breed_mutation(parents, children_per_parent, mutation_rate):
        mutation_children = []
        for parent in parents:
                for _ in range(children_per_parent):
                    new_child = copy.deepcopy(parent)
                    new_child.mutate(mutation_rate)
                    mutation_children.append(new_child)
        return np.array(mutation_children)

    def calculate_rewards(self):
        for solution in self.individuals:
            solution.calculate_reward()

    def find_best_individual(self):
        rewards = [i.reward for i in self.individuals]
        # Save the best one if better than current best
        if (self.best_individual is None or
                self.best_individual.reward < np.max(rewards)):
            self.best_individual = self.individuals[np.argmax(rewards)]

    @staticmethod
    def calculate_expected_weights_if_needed(gift_weight_init_method):
        if gift_weight_init_method in [
                GiftWeightInitMethod.expected_mean,
                GiftWeightInitMethod.expected_median,
                GiftWeightInitMethod.expected_analytical]:
            utils.set_expected_gift_weights(n_observations_per_gift=1000000,
                                            approach=gift_weight_init_method)

    @staticmethod
    def save_init_settings_on_hard_drive(
            population_size, gift_weight_init_method, gift_type_amounts):
        # Create parent folder if needed
        if not os.path.exists(SAVED_SOLUTIONS_FOLDER):
            os.makedirs(SAVED_SOLUTIONS_FOLDER)
        # Create child folder
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_folder = os.path.join(SAVED_SOLUTIONS_FOLDER, date_time)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        # Define output
        settings = [
            "\nInitialization settings\n-----------------------",
            "Population size: %s" % population_size,
            "Gift weight init method: %s" % gift_weight_init_method,
            "Gift type amounts to include: %s" % gift_type_amounts]
        # Create settings file, write to file & close the file
        f_out = open(os.path.join(results_folder, 'settings.txt'), 'w')
        f_out.write('\n'.join(settings))
        f_out.close()
        # Print the settings as well
        for s in settings:
            print(s)
        return results_folder

    def save_train_settings_on_hard_drive(
            self, n_parents, children_per_parent, n_generations, mutation_rate,
            selection_method):
        settings = [
            "\nTraining settings\n-----------------",
            "Nr of parents: %s" % n_parents,
            "Nr of children per parent: %s" % children_per_parent,
            "Nr of generations: %s" % n_generations,
            "Mutation rate: %s" % mutation_rate,
            "Selection method: %s" % selection_method,
            "Nr of bags: %s" % len(self.individuals[0].bags),
            "Max weight of a bag: %s" % utils.MAX_BAG_WEIGHT,
            "Minimum nr of gifts in a bag: %s" % utils.MIN_GIFTS_IN_BAG]
        f = open(os.path.join(self.results_folder_name, 'settings.txt'), "a")
        f.write('\n'.join(settings))
        f.close()
        # Print the settings as well
        for s in settings:
            print(s)


class SolutionCandidate:
    def __init__(self, gift_weight_init_method, gift_type_amounts):
        self.bags = self.initialize_bags(
            gift_weight_init_method, gift_type_amounts)
        self.reward = 0.0
        self.bags_expected_weights_std = 0.0
        self.calculate_reward()

    @staticmethod
    def initialize_bags(gift_weight_init_method, gift_type_amounts):
        gift_type_amounts_remaining = copy.deepcopy(gift_type_amounts)
        bags, gifts = [], []
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
                # Add gift only if its type hasn't been added enough already
                gift.initialize_weight(gift_weight_init_method)
                gifts.append(gift)
        # Randomize the order of gifts
        np.random.shuffle(gifts)
        # Put gifts in the bags
        all_bags_full = False
        while not all_bags_full:
            n_bags_filled = 0
            # For each bag, add the first occurring gift that fits in the bag
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
        # Calculate weights for all the bags
        for bag in bags:
            bag.calculate_weight()
        return bags

    def calculate_reward(self):
        rewards, reject_rate = [bag.simulate_weight(n_observations=100) for
                                bag in self.bags]
        self.std = rewards.std()
        self.reward = rewards.mean()

    def mutate(self, mutation_rate=0.001):
        if isinstance(mutation_rate, float) and 0. < mutation_rate < 1.:
            # Expecting that all gifts are in bags
            n_gifts_mutate = int(np.floor(mutation_rate * utils.N_GIFTS))
        else:
            n_gifts_mutate = int(mutation_rate)
        # Select the bags between which gifts are to be mutated (moved)
        bag_ix_from, bag_ix_to = self.get_mutation_bag_ix(n_gifts_mutate)
        # Move gifts between bags
        for bag_from, bag_to in zip(bag_ix_from, bag_ix_to):
            gift = np.random.choice(self.bags[bag_from].gifts, size=1)[0]
            self.bags[bag_to].add_gift(gift)
            self.bags[bag_to].calculate_weight()
            self.bags[bag_from].remove_gift(gift)
            self.bags[bag_from].calculate_weight()

    def get_mutation_bag_ix(self, n_gifts_mutate):
        bag_ix_from = np.random.choice(
            # There must be gift(s) in a bag
            [i for i, bag in enumerate(self.bags) if len(bag.gifts) > 1],
            size=n_gifts_mutate, replace=False)
        bag_ix_to = []
        for bag_ix in bag_ix_from:
            bag_ix_to.append(np.random.choice(
                # Let's not add gifts to bags where they were taken from
                [i for i in range(len(self.bags)) if i != bag_ix], size=1)[0])
        return bag_ix_from, np.array(bag_ix_to)

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


class Bag:
    def __init__(self, is_trash_bag=False):
        self.gifts = []
        self.weight = 0.0
        self.expected_weight = 0.0
        self.is_trash_bag = is_trash_bag

    def add_gift(self, gift):
        self.gifts.append(gift)
        self.weight += gift.weight

    def remove_gift(self, gift):
        self.gifts.remove(gift)
        self.weight -= gift.weight

    def calculate_weight(self):
        self.weight = np.array([gift.weight for gift in self.gifts]).sum()
        self.expected_weight = np.array(
            [SIMULATED_GIFTS[g.gift_type] for g in self.gifts]).sum(
            axis=0).mean()
        # print("Expected weight: %s" % self.expected_weight)

    def simulate_weight(self, n_observations=1000):
        bag_weights = []
        for _ in range(n_observations):
            gift_weigths = [
                utils.GIFT_WEIGHT_DISTRIBUTIONS[gift.gift_type.lower()]
                for gift in self.gifts]
            bag_weights.append(gift_weigths)
        reject_rate = (
            np.array([w > utils.MAX_BAG_WEIGHT for w in bag_weights]).sum() /
            len(bag_weights))

class Gift:
    def __init__(self, weight=None, id_label=None, gift_type=None):
        self.weight = weight
        self.id_label = id_label
        self.gift_type = gift_type

    def initialize_weight(self, gift_weight_init_method):
        if gift_weight_init_method in [
                GiftWeightInitMethod.expected_mean,
                GiftWeightInitMethod.expected_median,
                GiftWeightInitMethod.expected_analytical]:
            self.weight = utils.EXPECTED_GIFT_WEIGHTS[self.gift_type]
        elif gift_weight_init_method == GiftWeightInitMethod.sample_from_distr:
            self.weight = utils.GIFT_WEIGHT_DISTRIBUTIONS[
                self.gift_type.lower()]()
        else:
            raise (Exception("Unknown gift_weight_init_method"))
