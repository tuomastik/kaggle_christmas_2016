import geneticalgorithm

if __name__ == '__main__':

    ga = geneticalgorithm.GeneticAlgorithm(
        population_size=10,
        n_observations_to_evaluate_solution=100,
        gift_weight_init_method=(
            geneticalgorithm.GiftWeightInitMethod.sample_from_distr),
        gift_type_amounts={
            'Horse': 1000,    # Max 1000
            'Ball': 1100,     # Max 1100
            'Bike': 50,       # Max 500
            'Train': 1000,    # Max 1000
            'Coal': 0,        # Max 166
            'Book': 1200,     # Max 1200
            'Doll': 1000,     # Max 1000
            'Blocks': 1000,   # Max 1000
            'Gloves': 200})   # Max 200

    ga.train(n_generations=10000, for_reproduction=0.2, mutation_rate=1,
             selection_method=geneticalgorithm.SelectionMethod.truncation)
