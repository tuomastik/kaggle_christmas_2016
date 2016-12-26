import os
import datetime

import numpy as np

SUBMISSIONS_FOLDER = 'submissions'
submission_nr = 1

# Import solution
solution = np.load(os.path.join(
    'ga_solutions', 'best_individual-reward_'
    '42101.8089799-2016-12-26_01-08-06.npy')).item()['best_individual']

if not os.path.exists(SUBMISSIONS_FOLDER):
    os.makedirs(SUBMISSIONS_FOLDER)

# Create submission file
f_out = open(os.path.join(SUBMISSIONS_FOLDER, '%s_submission_%s.csv' % (
    str(submission_nr).zfill(2),
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'w')
f_out.write('Gifts\n')
# Go through gifts in the bags
for i, bag in enumerate(solution.bags, start=1):
    print('\n\nBag  %-6s Weight %15s' % (i, 'Label'))
    print('----------------------------------')
    for j, gift in enumerate(bag.gifts, start=1):
        print('Gift %-6s %6.2f %15s' % (j, gift.weight, gift.id_label))
    print('----------------------------------')
    print('Total %12.2f' % bag.weight)
    f_out.write(' '.join([g.id_label for g in bag.gifts]) + '\n')
f_out.close()
