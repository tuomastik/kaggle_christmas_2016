from scipy.optimize import basinhopping
import numpy as np

import utils


gifts_type_counts = utils.GIFTS_DF['GiftId'].apply(
    lambda x: x.split('_')[0]).value_counts()

# Type of each gift
gift_types = []
for gift_type, gift_count in gifts_type_counts.to_dict().items():
    gift_types += [gift_type] * gift_count

# Bag of each gift
gift_bags_initial = [np.random.choice(utils.N_BAGS) for _ in
                     range(len(gift_types))]


def function_to_minimize(gift_bags):
    # print(gift_bags.sum())
    # Convert to integer because the values represent bags and
    # the values are used to index a list (doesn't work with floats)
    gift_bags = np.array(gift_bags).astype(int)
    # Gather together weights of gifts in each bag
    bags_gifts_weights = [[] for _ in range(utils.N_BAGS)]
    for i, (bag_ix, gift_type) in enumerate(zip(gift_bags, gift_types)):
        # Sample new weight every time
        # bags_gifts_weights[bag_ix].append(
        #     utils.GIFT_WEIGHT_DISTRIBUTIONS[gift_type]())
        # Use pre calculated weights
        bags_gifts_weights[bag_ix].append(
            utils.SIMULATED_GIFTS[gift_type.title()][i])
    # Sum weights of gifts in each bag
    for i, bag_gifts_weights in enumerate(bags_gifts_weights):
        bag_weight = sum(bag_gifts_weights)
        gifts_in_bag = len(bag_gifts_weights)
        if gifts_in_bag >= 3 and bag_weight <= utils.MAX_BAG_WEIGHT:
            bags_gifts_weights[i] = bag_weight
        else:
            bags_gifts_weights[i] = 0
    # We want to maximize the following
    weight_of_bags = sum(bags_gifts_weights)
    print("Score: %.3f" % weight_of_bags)
    # However, SciPy optimizer wants to minimize the function.
    # A maximization is the minimization of the -1*function.
    return -1 * weight_of_bags


bh = basinhopping(function_to_minimize, np.array(gift_bags_initial),
                  minimizer_kwargs={"method": "L-BFGS-B"}, niter=1,
                  disp=True, stepsize=1.0)
print("global minimum: x = %.4f, f(x0) = %.4f" % (bh.x, bh.fun))


# -----------------------
# Minimal working example
# -----------------------

from scipy.optimize import basinhopping
import numpy as np

def func2d(x):
    print(x)
    f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f

minimizer_kwargs = {"method":"L-BFGS-B"}
x0 = [1.0, 1.0]
ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs, niter=1,
                   stepsize=.5)
