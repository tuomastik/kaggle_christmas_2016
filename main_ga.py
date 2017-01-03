import os

import geneticalgorithm

if __name__ == '__main__':

    ga = geneticalgorithm.GeneticAlgorithm(
        population_size=100,
        n_observations_to_evaluate_solution=1000,
        gift_weight_init_method=(
            geneticalgorithm.GiftWeightInitMethod.sample_from_distr),
        warm_start_path=os.path.join(
            'ga_solutions', '2017-01-01_18-01-18',
            '2017-01-02_19-37-54--reward=32275.3248013--submission.csv'),
        gift_type_amounts={
            'Horse': 1000,    # Max 1000
            'Ball': 1100,     # Max 1100
            'Bike': 0,        # Max 500
            'Train': 1000,    # Max 1000
            'Coal': 0,        # Max 166
            'Book': 1200,     # Max 1200
            'Doll': 1000,     # Max 1000
            'Blocks': 1000,   # Max 1000
            'Gloves': 200})   # Max 200

    ga.train(n_generations=10000, for_reproduction=0.04, mutation_rate=2,
             selection_method=geneticalgorithm.SelectionMethod.truncation,
             swap_proba=0.5)
