import utils


class GiftWeightInitMethod:
    def __init__(self):
        pass
    expected_mean = 'numerical_mean'
    expected_median = 'numerical_median'
    expected_analytical = 'analytical'
    sample_from_distr = 'sample_from_distribution'


class Gift:
    def __init__(self, weight=None, id_label=None, gift_type=None):
        self.weight = weight
        self.id_label = id_label
        self.gift_type = gift_type

    def initialize_weight(self, gift_weight_init_method):
        if gift_weight_init_method in [
                GiftWeightInitMethod.expected_mean,
                GiftWeightInitMethod.expected_median,
                GiftWeightInitMethod.expected_analytical]:
            self.weight = utils.EXPECTED_GIFT_WEIGHTS[self.gift_type]
        elif gift_weight_init_method == GiftWeightInitMethod.sample_from_distr:
            self.weight = utils.GIFT_WEIGHT_DISTRIBUTIONS[
                self.gift_type.lower()]()
        else:
            raise (Exception("Unknown gift_weight_init_method"))
