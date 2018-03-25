import evolutionary_algorithm

if __name__ == '__main__':

    ea = evolutionary_algorithm.EvolutionaryAlgorithm(
        population_size=100,
        n_observations_to_evaluate_solution=100,
        gift_weight_init_method=(
            evolutionary_algorithm.GiftWeightInitMethod.sample_from_distr),
        warm_start_path=None,
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

    # Debugging
    # ea.individuals[0].print_expected_weights()
    # ea.individuals[0].get_gift_type_counts()
    # ea.individuals[0].shuffle_gift_ids(gift_type_to_shuffle='bike')
    # ea.individuals[0].print_expected_weights()
    # ea.individuals[0].save_on_hard_drive(ea.results_folder_name)

    ea.train(n_generations=10000, for_reproduction=0.2, mutation_rate=1,
             selection_method=evolutionary_algorithm.SelectionMethod.truncation,
             swap_proba=0.5)
