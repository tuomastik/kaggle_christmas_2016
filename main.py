import numpy as np

import utils
import geneticalgorithm

if __name__ == '__main__':
    seed = 1122345
    np.random.seed(seed)

    utils.set_expected_gift_weights(n_observations_per_gift=1000000,
                                    approach='numerical_median')
    ga = geneticalgorithm.GeneticAlgorithm(population_size=100)
    ga.train(n_generations=10000, for_reproduction=0.05)
