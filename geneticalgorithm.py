import os
import copy
import datetime

import numpy as np

import utils


GIFTS_PER_BAG_INITIALLY = np.diff(np.linspace(
    start=0, stop=utils.N_GIFTS, num=utils.N_BAGS + 1, dtype=int))


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


class SolutionInitMethod:
    def __init__(self):
        pass
    random = 'random'
    greedy = 'greedy'


class GeneticAlgorithm:
    SAVED_SOLUTIONS_FOLDER = 'ga_solutions'

    def __init__(self, population_size=50,
                 gift_weight_init_method=GiftWeightInitMethod.expected_mean,
                 solution_init_method=SolutionInitMethod.greedy):
        print("\nSettings\n--------")
        print("Population size: %s" % population_size)
        print("Gift weight init method: %s" % gift_weight_init_method)
        print("Solution init method: %s\n" % solution_init_method)
        self.calculate_expected_weights_if_needed(gift_weight_init_method)
        self.individuals = []
        for i in range(1, population_size + 1):
            print("\rInitializing population - solution candidate %s / %s" % (
                  i, population_size), end='', flush=True)  # Print same line
            self.individuals.append(SolutionCandidate(gift_weight_init_method,
                                                      solution_init_method))
        self.individuals = np.array(self.individuals)
        self.best_individual = None

    def train(self, n_generations=1000, for_reproduction=0.1,
              mutation_rate=0.01, selection_method=SelectionMethod.truncation,
              n_generations_weight_resample=None):
        print("\nStarting training...")
        # Pre calculate things.
        # Round the number of parents to be selected to an even number so that
        # we can execute bio-inspired operators in a pair wise manner.
        n_parents = utils.round_down_to_even(for_reproduction *
                                             len(self.individuals))
        children_per_parent = int(1.0 / for_reproduction)

        # Print settings
        print("\nSettings\n--------")
        print("Nr of parents: %s" % n_parents)
        print("Nr of children per parent: %s" % children_per_parent)
        print("Nr of generations: %s" % n_generations)
        print("Mutation rate: %s" % mutation_rate)
        print("Selection method: %s" % selection_method)
        print("Resample gift weights after generations: %s" % (
            n_generations_weight_resample))
        print("Nr of bags: %s" % utils.N_BAGS)
        print("Max weight of a bag: %s" % utils.MAX_BAG_WEIGHT)
        print("Minimum nr of gifts in a bag: %s" % utils.MIN_GIFTS_IN_BAG)

        # Evolution
        for generation in range(1, n_generations + 1):

            parents = self.select_parents(selection_method, n_parents)

            if (n_generations_weight_resample is not None and
                    generation % n_generations_weight_resample == 0):
                print("\nResampling gift weights for the new parents")
                parents = self.resample_weights(parents)

            print("\nGeneration %s" % generation)
            print("Rewards of the new parents\n--------------------------")
            for p in parents:
                print("%.3f" % p.reward)

            self.individuals = self.breed_mutation(
                parents=parents, children_per_parent=children_per_parent,
                mutation_rate=mutation_rate)
            self.calculate_rewards()

            self.find_best_individual()
            if generation % 10 == 0:
                # Every 10th generation
                self.save_best_individual_on_hard_drive()
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

    @staticmethod
    def calculate_expected_weights_if_needed(gift_weight_init_method):
        if gift_weight_init_method in [
                GiftWeightInitMethod.expected_mean,
                GiftWeightInitMethod.expected_median,
                GiftWeightInitMethod.expected_analytical]:
            utils.set_expected_gift_weights(n_observations_per_gift=1000000,
                                            approach=gift_weight_init_method)

    @staticmethod
    def resample_weights(individuals):
        for i, individual in enumerate(individuals):
            for j, bag in enumerate(individual.bags):
                for k, gift in enumerate(bag.gifts):
                    individuals[i].bags[j].gifts[k].weight = (
                        utils.GIFT_WEIGHT_DISTRIBUTIONS[
                            gift.gift_type.lower()]())
                individuals[i].bags[j].calculate_weight()
            individuals[i].calculate_reward()
        return individuals


class SolutionCandidate:
    def __init__(self, gift_weight_init_method, solution_init_method):
        self.bags = self.initialize_bags(gift_weight_init_method,
                                         solution_init_method)
        self.reward = 0.0
        self.calculate_reward()

    @staticmethod
    def initialize_bags_random(gift_weight_init_method):
        bags = []
        random_row_indices = np.random.permutation(utils.N_GIFTS)
        for n_gifts in GIFTS_PER_BAG_INITIALLY:
            bag = Bag(is_trash_bag=False)
            extracted_gifts = utils.GIFTS_DF.iloc[random_row_indices[:n_gifts]]
            random_row_indices = random_row_indices[n_gifts:]
            for row_ix, gift in extracted_gifts.iterrows():
                gift = Gift(id_label=gift['GiftId'],
                            gift_type=gift['GiftId'].split('_')[0].title())
                gift.initialize_weight(gift_weight_init_method)
                bag.add_gift(gift=gift)
            bag.calculate_weight()
            bags.append(bag)
        # Add one trash bag for gifts not included in the solution reward
        bags.append(Bag(is_trash_bag=True))
        return bags

    @staticmethod
    def initialize_bags_greedy(gift_weight_init_method):
        bags, gifts = [], []
        # Initialize gifts
        for _, row in utils.GIFTS_DF.iterrows():
            gift = Gift(id_label=row['GiftId'],
                        gift_type=row['GiftId'].split('_')[0].title())
            gift.initialize_weight(gift_weight_init_method)
            gifts.append(gift)
        # Sort gifts based on weight (first gift will be the heaviest)
        gifts.sort(key=lambda x: x.weight, reverse=True)
        # Put gifts in the bags
        for gifts_per_bag in GIFTS_PER_BAG_INITIALLY:
            bag = Bag(is_trash_bag=False)
            added_gifts = []
            for gift in gifts:
                if (len(bag.gifts) < gifts_per_bag and
                        bag.weight + gift.weight < utils.MAX_BAG_WEIGHT):
                    bag.add_gift(gift)
                    added_gifts.append(gift)
            bags.append(bag)
            # Remove added gifts from list of gifts
            gifts = [g for g in gifts if g not in added_gifts]
        if len(gifts) > 0:
            # Add the remaining gifts to random bags
            # print("Randomly assigning a bag for each %s leftover gift" %
            #       len(gifts))
            for bag_ix, gift in zip(
                    np.random.randint(low=0, high=utils.N_BAGS, size=len(gifts)),
                    gifts):
                bags[bag_ix].add_gift(gift)
        # Calculate weights for all the bags
        for bag in bags:
            bag.calculate_weight()
        # Add one trash bag for gifts not included in the solution reward
        bags.append(Bag(is_trash_bag=True))
        return bags

    @staticmethod
    def initialize_bags(gift_weight_init_method, solution_init_method):
        if solution_init_method == SolutionInitMethod.random:
            return SolutionCandidate.initialize_bags_random(
                gift_weight_init_method)
        elif solution_init_method == SolutionInitMethod.greedy:
            return SolutionCandidate.initialize_bags_greedy(
                gift_weight_init_method)
        else:
            raise(Exception("Unknown solution_init_method"))

    def calculate_reward(self):
        self.reward = 0.0
        for bag in self.bags:
            if (not bag.is_trash_bag and len(bag.gifts) >= 3 and
                    bag.expected_weight <= utils.MAX_BAG_WEIGHT):
                self.reward += bag.expected_weight
            # else:
            #     print("  - Discarding bag because there are too few gifts "
            #           "in the bag or the bag is too heavy!")

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
                [i for i in range(utils.N_BAGS) if i != bag_ix], size=1)[0])
        return bag_ix_from, np.array(bag_ix_to)


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
