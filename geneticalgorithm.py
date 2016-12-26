import os
import copy
import datetime

import numpy as np

import utils


GIFTS_PER_BAG_INITIALLY = np.diff(np.linspace(
    start=0, stop=utils.N_GIFTS, num=utils.N_BAGS + 1, dtype=int))


class GeneticAlgorithm:
    SAVED_SOLUTIONS_FOLDER = 'ga_solutions'

    def __init__(self, population_size=50):
        self.individuals = np.array([SolutionCandidate() for _ in
                                     range(population_size)])
        self.best_individual = None

    def train(self, n_generations=1000, for_reproduction=0.1,
              mutation_rate=0.01):
        # Round the number of parents to be selected to an even number so that
        # we can execute bio-inspired operators in a pair wise manner.
        n_parents = utils.round_down_to_even(for_reproduction *
                                             len(self.individuals))
        children_per_parent = int(1.0 / for_reproduction)
        for generation in range(1, n_generations + 1):
            # fittest = self.parent_selection_roulette_wheel(n_parents)
            fittest = self.parent_selection_truncation(n_parents)

            print("")
            print("Rewards of the new parents")
            print("--------------------------")
            for f in fittest:
                print("%.3f" % f.reward)

            self.individuals = self.breed_mutation(
                parents=fittest, children_per_parent=children_per_parent,
                mutation_rate=mutation_rate)
            self.calculate_rewards()
            self.find_best_individual()
            if generation % 10 == 0:
                # Every 10th generation
                self.save_best_individual_on_hard_drive()

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
        # Find indices of n_best solution rewards
        best_ix = np.argpartition(rewards, -n_parents)[-n_parents:]
        fittest_solutions = self.individuals[best_ix]
        return fittest_solutions

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

    def save_best_individual_on_hard_drive(self):
        # Create output folder if it does not exist yet
        if not os.path.exists(self.SAVED_SOLUTIONS_FOLDER):
            os.makedirs(self.SAVED_SOLUTIONS_FOLDER)

        date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_path = os.path.join(
            self.SAVED_SOLUTIONS_FOLDER,
            'best_individual-reward_%s-%s.npy' % (self.best_individual.reward,
                                                  date_time))
        variables_save = {'best_individual': self.best_individual}
        np.save(file_path, variables_save)


class SolutionCandidate:
    def __init__(self):
        # print("Creating solution...")
        self.bags = self.initialize_bags()
        self.reward = 0
        self.calculate_reward()
        # print("  - Reward: %s" % self.reward)

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
                bag.add_new_gift(
                    weight=utils.EXPECTED_GIFT_WEIGHTS[gift['gift_type']],
                    id_label=gift['GiftId'],
                    running_number=row_ix)
            bags.append(bag)
        return bags

    def calculate_reward(self):
        self.reward = 0
        for bag in self.bags:
            if len(bag.gifts) >= 3 and bag.weight <= utils.MAX_BAG_WEIGHT:
                self.reward += bag.weight
            # else:
            #     print("  - Discarding bag because there are too few gifts "
            #           "in the bag or the bag is too heavy!")

    def mutate(self, mutation_rate=0.001):
        if isinstance(mutation_rate, float) and 0 <= mutation_rate <= 1:
            # Expecting that all gifts are in bags
            n_gifts_mutate = int(np.floor(mutation_rate * utils.N_GIFTS))
        else:
            n_gifts_mutate = int(mutation_rate)
        # Select the bags between which gifts are to be mutated (moved)
        bag_ix_from, bag_ix_to = self.get_mutation_bag_ix(n_gifts_mutate)
        # Move gifts between bags
        for bag_from, bag_to in zip(bag_ix_from, bag_ix_to):
            gift = np.random.choice(self.bags[bag_from].gifts, size=1)[0]
            self.bags[bag_to].add_existing_gift(gift)
            self.bags[bag_from].remove_gift(gift)

    def get_mutation_bag_ix(self, n_gifts_mutate):
        # Appreciate heavier bags more when deciding the bags from which
        # gifts will be moved away
        bag_weights = np.array([b.weight for b in self.bags])
        bag_weights_probas = bag_weights / bag_weights.sum()
        bag_ix_from = np.random.choice(
            range(utils.N_BAGS), size=n_gifts_mutate, replace=False,
            p=bag_weights_probas)
        # Appreciate lighter bags more when deciding the bags in which
        # gifts will be moved to
        bag_ix_available, bag_weights_available = [], []
        for i in range(utils.N_BAGS):
            if i not in bag_ix_from:
                bag_ix_available.append(i)
                bag_weights_available.append(1./bag_weights[i])
        bag_weights_available = np.array(bag_weights_available)
        bag_weights_available_probas = (bag_weights_available /
                                        bag_weights_available.sum())
        bag_ix_to = np.random.choice(
            bag_ix_available, size=n_gifts_mutate, replace=False,
            p=bag_weights_available_probas)
        return bag_ix_from, bag_ix_to


class Bag:
    def __init__(self):
        self.gifts = []
        self.weight = 0.0

    def add_new_gift(self, weight, id_label, running_number):
        self.gifts.append(Gift(weight, id_label, running_number))
        self.weight += weight

    def add_existing_gift(self, gift):
        self.gifts.append(gift)
        self.weight += gift.weight

    def remove_gift(self, gift):
        self.gifts.remove(gift)
        self.weight -= gift.weight


class Gift:
    def __init__(self, weight, id_label, running_nr):
        self.weight = weight
        self.id_label = id_label
        self.running_nr = running_nr
