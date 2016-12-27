import numpy as np

import geneticalgorithm

if __name__ == '__main__':
    seed = 1122345
    np.random.seed(seed)
    ga = geneticalgorithm.GeneticAlgorithm(
        population_size=200, gift_weight_init_method=(
            geneticalgorithm.GiftWeightInitMethod.sample_from_distr))
    ga.train(n_generations=10000, for_reproduction=0.1, mutation_rate=20,
             n_generations_weight_resample=1)
