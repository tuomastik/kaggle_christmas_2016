import os
import copy
from datetime import datetime

import numpy as np

import utils
from solutioncandidate import SolutionCandidate
from gift import GiftWeightInitMethod


SAVED_SOLUTIONS_FOLDER = 'ga_solutions'


class SelectionMethod:
    def __init__(self):
        pass
    truncation = 'truncation'
    roulette_wheel = 'roulette_wheel'


class GeneticAlgorithm:

    def __init__(self, population_size=50,
                 n_observations_to_evaluate_solution=100,
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
                gift_weight_init_method, gift_type_amounts,
                n_observations_to_evaluate_solution))
        self.individuals = np.array(self.individuals)
        self.best_individual = None

    def train(self, n_generations=1000, for_reproduction=0.1,
              mutation_rate=0.01, selection_method=SelectionMethod.truncation):
        print("\nStarting training...")

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
                print("Reward: %.3f (std: %.3f) - Mean reject rate: %.3f" %
                      (p.reward, p.reward_std, p.mean_reject_rate))

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
        rewards = np.array([i.reward for i in self.individuals])
        uniques, unique_ix = np.unique(rewards, return_index=True)
        if len(uniques) >= n_parents:
            # Select unique best solutions if enough available
            best_uniques_ix = np.argpartition(uniques, -n_parents)[-n_parents:]
            best_unique_rewards_ix = unique_ix[best_uniques_ix]
            parents = self.individuals[best_unique_rewards_ix]
        else:
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
            solution.calculate_reward(
                solution.n_observations_to_evaluate_solution)

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
