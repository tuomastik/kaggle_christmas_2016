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

    def train(self, n_generations=10, for_reproduction=0.1):
        for _ in range(n_generations):
            fittest = self.get_fittest_individuals(for_reproduction)

            print("")
            print("Rewards of the fittest")
            print("----------------------")
            for f in fittest:
                print("%.3f" % f.reward)

            self.individuals = self.breed_mutation(
                parents=fittest, children_per_parent=int(1.0/for_reproduction))
            self.calculate_rewards()
            self.save_best_individual_on_hard_drive()

    def get_fittest_individuals(self, for_reproduction):
        rewards = [i.reward for i in self.individuals]
        # Save the best one if better than current best
        if (self.best_individual is None or
                self.best_individual.reward < np.max(rewards)):
            self.best_individual = self.individuals[np.argmax(rewards)]
        # Round the number of fittest individuals to be selected to
        # an even number so that we can execute bio-inspired operators
        # in a pair wise manner.
        n_best = utils.round_down_to_even(for_reproduction * len(rewards))
        # Find indices of n_best solution rewards
        best_ix = np.argpartition(rewards, -n_best)[-n_best:]
        fittest_solutions = self.individuals[best_ix]
        return fittest_solutions

    @staticmethod
    def breed_mutation(parents, children_per_parent):
        mutation_children = []
        for parent in parents:
                for _ in range(children_per_parent):
                    new_child = copy.deepcopy(parent)
                    new_child.mutate()
                    mutation_children.append(new_child)
        return np.array(mutation_children)

    def calculate_rewards(self):
        for solution in self.individuals:
            solution.calculate_reward()

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
            if bag.weight <= utils.MAX_BAG_WEIGHT:
                self.reward += bag.weight
            # else:
            #     print("  - Too heavy bag!")

    def mutate(self, mutation_rate=0.001):
        # Expecting that all gifts are in bags
        n_gifts_mutate = int(np.floor(mutation_rate * utils.N_GIFTS))
        # Determine randomly which gifts to mutate
        gift_ix_mutate = np.random.permutation(utils.N_GIFTS)[:n_gifts_mutate]
        # First, delete those gifts from bags
        old_bag_ix, deleted_gifts = self.delete_gifts(gift_ix=gift_ix_mutate)
        # Determine randomly in which different bag to add the gifts
        new_bag_ix = [i for i in np.random.randint(
            low=0, high=utils.N_BAGS, size=n_gifts_mutate + len(old_bag_ix))
                      if i not in old_bag_ix][:n_gifts_mutate]
        self.add_gifts(gifts=deleted_gifts, bag_indices=new_bag_ix)

    def delete_gifts(self, gift_ix):
        # Store the indices of bags where the gifts are found.
        # Store the gifts that are deleted.
        bag_ix, deleted_gifts = [], []
        for i, bag in enumerate(self.bags):
            gifts_to_remove = []
            for gift in bag.gifts:
                if gift.running_nr in gift_ix:
                    bag_ix.append(i)
                    deleted_gifts.append(gift)
                    gifts_to_remove.append(gift)
            for gift_to_remove in gifts_to_remove:
                bag.remove_gift(gift_to_remove)
        return bag_ix, deleted_gifts

    def add_gifts(self, gifts, bag_indices):
        for bag_ix, gift in zip(bag_indices, gifts):
            self.bags[bag_ix].gifts.append(gift)


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
