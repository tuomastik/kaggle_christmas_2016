from scipy.optimize import basinhopping
import numpy as np

import utils

gifts_type_counts = utils.GIFTS_DF['GiftId'].apply(
    lambda x: x.split('_')[0]).value_counts()

# Type of each gift (7166 long)
gift_types = []
for gift_type, gift_count in gifts_type_counts.to_dict().items():
    gift_types += [gift_type] * gift_count

unique_gift_types = list(set(gift_types))
gift_type_weights = utils.SIMULATED_GIFTS.mean(axis=0)

# initialize bag_gift_matrix
num_bags_to_minimize = 20
bag_gift_mat_shape = (len(unique_gift_types), num_bags_to_minimize)
print(bag_gift_mat_shape)

# bag_gift_mat = np.zeros((bag_gift_mat_shape))
bag_gift_mat = np.random.rand(bag_gift_mat_shape[0], bag_gift_mat_shape[1])


def limiting_sigmoid(x, steepness=1):
    return 1.0 - 1.0 / (1 + np.exp(-steepness * (x - 50)))


def func(bag_gift_matrix):
    bag_gift_matrix = bag_gift_matrix.reshape(bag_gift_mat_shape)
    bag_weights = np.dot(bag_gift_matrix.transpose(), gift_type_weights)
    # limit bag weights to 50kg smoothly by using sigmoid function
    bag_weights = [limiting_sigmoid(bw, steepness=1) for bw in bag_weights]
    bag_weights = np.array(bag_weights)

    print("Num gifts: {} - sum: {}".format(bag_gift_matrix.sum(), bag_weights.sum()))
    return -bag_weights.sum()

#
def bounds(f_new, x_new, f_old, x_old):
    is_positive = (x_new > 0).all()
    return is_positive

optimizer_bounds = [(0, None) for _ in range(np.prod(bag_gift_mat_shape))]

bh = basinhopping(func,
                  bag_gift_mat,
                  minimizer_kwargs={"method": "L-BFGS-B", "bounds": optimizer_bounds},
                  niter=100,
                  # accept_test=bounds,
                  disp=True,
                  stepsize=1.0)

print("Found minimum: f(x0) = {:.4f}".format(bh.fun))
print(bh.x)
