from scipy.optimize import basinhopping
import numpy as np

import utils

# how many bags are optimized at once
num_bags_to_minimize = 200
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

    # add tikhonov regularizer
    # (this tries to put every gift type into each bag and not use alot
    # any single type of gift)
    # alpha = 10 # how strongly this regularizer is considered
    # tikhonov = (matrix ** 2).sum()

    # add integer regularizer
    # as the total_bag_weight increases integer numbers are more preferred
    beta = 100 * total_bag_weight / 30000
    integer_regularizer = np.abs(
        np.round(matrix) - matrix).sum()
    return -total_bag_weight.sum() + beta * integer_regularizer


def limiting_sigmoid(x, steepness=1):
    return 1.0 - 1.0 / (1 + np.exp(-steepness * (x - 50)))


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
    has_3_or_more = x_new.sum(axis=0) >= 3
    num_gfts_allowed = gifts_type_counts * bag_batch_ratio
    gift_count_limits_exceeded = x_new.sum(axis=1) > num_gfts_allowed
    return has_3_or_more.all() and not gift_count_limits_exceeded.any()


def main():
    # initialize matrix with random numbers between 0 - 1
    matrix = np.random.rand(matrix_shape[0], matrix_shape[1])

    # set optimizer bound to disallow negative gift counts
    optimizer_bounds = [(0, None) for _ in range(np.prod(matrix_shape))]
    print("Begin hopping..")
    bh = basinhopping(func,
                      matrix,
                      minimizer_kwargs={"method": "L-BFGS-B",
                                        "bounds": optimizer_bounds},
                      niter=1,
                      accept_test=accept_func,
                      disp=True,
                      stepsize=1.0)

    print("Found minimum: f(x0) = {:.4f}".format(bh.fun))
    print("Weight per bag: {:.4f}".format(-bh.fun / num_bags_to_minimize))
    print(bh.x)
    mat = bh.x.reshape(matrix_shape)
    gift_counts = mat.astype(int).sum(axis=1)
    print("Bag gift counts \n{}".format(gift_counts.transpose()))
    print("Total gift counts \n{}".format(gifts_type_counts))

    # TODO: iteratively decrement total gift count
    # TODO: try different minimizers
    # TODO: Tee palautus
    # TODO: gift counts ad pandas dataframe


if __name__ == '__main__':
    main()
