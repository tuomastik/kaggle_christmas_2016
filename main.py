import numpy as np

import geneticalgorithm

if __name__ == '__main__':
    seed = 1122345
    np.random.seed(seed)
    ga = geneticalgorithm.GeneticAlgorithm(
        population_size=100,
        gift_weight_init_method=(
            geneticalgorithm.GiftWeightInitMethod.sample_from_distr),
        gift_type_amounts={
            'Horse': 1000,    # Max 1000
            'Ball': 1100,     # Max 1100
            'Bike': 100,      # Max 500
            'Train': 1000,    # Max 1000
            'Coal': 0,        # Max 166
            'Book': 1200,     # Max 1200
            'Doll': 1000,     # Max 1000
            'Blocks': 1000,   # Max 1000
            'Gloves': 200})   # Max 200
    ga.train(n_generations=10000, for_reproduction=0.04, mutation_rate=1,
             selection_method=geneticalgorithm.SelectionMethod.roulette_wheel)
