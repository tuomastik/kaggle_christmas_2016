import os

import geneticalgorithm

if __name__ == '__main__':

    ga = geneticalgorithm.GeneticAlgorithm(
        population_size=10,
        n_observations_to_evaluate_solution=1000,
        gift_weight_init_method=(
            geneticalgorithm.GiftWeightInitMethod.sample_from_distr),
        warm_start_path=os.path.join(
            'ga_solutions', '2016-12-29_13-55-53',
            '2016-12-30_13-25-10--reward=38627.6564727--submission.csv'),
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

    ga.train(n_generations=10000, for_reproduction=0.4, mutation_rate=1,
             selection_method=geneticalgorithm.SelectionMethod.truncation)
