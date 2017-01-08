import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import numpy as np
import pandas as pd

import utils

# how many bags are optimized at once
num_bags_to_minimize = 5
bag_batch_ratio = num_bags_to_minimize / 1000

gifts_type_counts = utils.GIFTS_DF['GiftId'].apply(
    lambda x: x.split('_')[0]).value_counts()

# Type of each gift (7166 long)
gift_types = []
for gift_type, gift_count in gifts_type_counts.to_dict().items():
    gift_types += [gift_type] * gift_count

unique_gift_types = list(set(gift_types))
gift_type_weights = utils.SIMULATED_GIFTS.mean(axis=0)

# shape of the parameter matrix
matrix_shape = (len(unique_gift_types), num_bags_to_minimize)


def func(matrix):
    total_bag_weight = compute_bag_weights_sigmoid(matrix,
                                                   gift_type_weights,
                                                   sigmoid_steepness=0.5)
    # print("Num gifts: {} - sum: {} - weight per bag: {}"
    #       .format(matrix.sum(), total_bag_weight.sum(),
    #               total_bag_weight.sum() / num_bags_to_minimize))

    # ADD TIKHONOV REGULARIZER
    # (this tries to put every gift type into each bag and not use alot
    # any single type of gift)
    alpha = 10 # how strongly this regularizer is considered
    tikhonov = (matrix ** 2).sum()

    # ADD INTEGER REGULARIZER
    integer_regularizer = np.abs(np.round(matrix) - matrix).sum()
    # as the total_bag_weight increases integer numbers are valued even more
    # BETA OPTION 1:  linear increase in regularization as the total weight in bags increases
    # beta = 30 * total_bag_weight / (num_bags_to_minimize * 50)
    # BETA OPTION 2:  nonlinear increase in regularization as the total weight in bags increases
    # beta = 10 ** 2 * exponential_function(total_bag_weight,
    #                                        high_at=num_bags_to_minimize * 50,
    #                                        shape=4)
    # BETA OPTION 3:  nonlinear increase in regularization as the total weight in bags increases
    beta = 100 * limiting_sigmoid(total_bag_weight, steepness=-0.05, limit=num_bags_to_minimize*50)
    return -total_bag_weight.sum() + beta * integer_regularizer


def limiting_sigmoid(x, steepness=1, limit=50):
    return 1.0 - 1.0 / (1 + np.exp(-steepness * (x - limit)))


def exponential_function(x, high_at, shape=3):
    return np.exp((x - high_at) / shape + shape)


def compute_bag_weights_sigmoid(matrix, gift_type_weights,
                                sigmoid_steepness=2):
    # reshape back to matrix shape, for some reason x_new is always 1d array
    matrix = matrix.reshape(matrix_shape)
    bag_weights = np.dot(matrix.transpose(), gift_type_weights)
    # limit bag weights to 50kg smoothly by using sigmoid function
    bag_weights = [bw * limiting_sigmoid(bw, sigmoid_steepness) for bw in
                   bag_weights]
    bag_weights = np.array(bag_weights)
    return bag_weights.sum()


def accept_func(f_new, x_new, f_old, x_old):
    # reshape back to matrix shape, for some reason x_new is always 1d array
    x_new = x_new.reshape(matrix_shape)
    return check_has_3_or_more(x_new).all() and check_is_positive(x_new).all()


def check_has_3_or_more(matrix):
    return matrix.sum(axis=0) >= 3


def check_is_positive(matrix):
    return matrix >= 0


def print_results(basinhopping_solution):
    print("Found minimum: f(x0) = {:.4f}".format(basinhopping_solution.fun))
    mat = basinhopping_solution.x.reshape(matrix_shape)
    mat_int = mat.round().astype(int)
    np.set_printoptions(precision=4, suppress=True)
    print("Float matrix: {}".format(mat))
    float_weight = compute_bag_weights_sigmoid(mat, gift_type_weights,
                                               sigmoid_steepness=20)
    int_weight = compute_bag_weights_sigmoid(mat.round().astype(int),
                                             gift_type_weights,
                                             sigmoid_steepness=20)
    print(
        "Forcing solution as integers with steep sigmoid:\n"
        "     Float sum: {} - Int sum: {}".format( float_weight, int_weight))
    print("Weight per bag:\n    Float: {:.4f} - Int: {:.4f}".format(
        float_weight / num_bags_to_minimize,
        int_weight / num_bags_to_minimize))
    print("Is solution acceptable: {}".format(
        accept_func(mat_int, mat_int, mat_int, mat_int)))
    print("    All gifts have positive amount: {}".format(
        check_is_positive(mat_int).all()))
    print("    Bags have 3 or more gifts: {}".format(
        check_has_3_or_more(mat_int).all()))

    # gift counts
    gift_counts = pd.DataFrame(mat.astype(int).sum(axis=1),
                               index=gifts_type_counts.index)
    print("Bag gift counts \n{}".format(gift_counts))
    print("Total gift counts \n{}".format(gifts_type_counts))



def main():
    # initialize matrix with random numbers between 0 - 1
    init_matrix = np.random.rand(matrix_shape[0], matrix_shape[1])

    # set optimizer bound to disallow negative gift counts
    optimizer_bounds = [(0, None) for _ in range(np.prod(matrix_shape))]
    print("Begin hopping..")
    bh = basinhopping(func,
                      init_matrix,
                      minimizer_kwargs={"method": "L-BFGS-B", "bounds": optimizer_bounds},
                      # minimizer_kwargs={"method": "TNC", "bounds": optimizer_bounds},
                      # minimizer_kwargs={"method": "COBYLA", "bounds": optimizer_bounds},
                      # minimizer_kwargs={"method": "SLSQP", "bounds": optimizer_bounds},
                      niter=1000,
                      # accept_test=accept_func,
                      disp=True,
                      stepsize=2)

    print_results(bh)

    # TODO: try different minimizers
    # TODO: Iteratively optimize all bags
    # TODO: Make submission


if __name__ == '__main__':
    main()
