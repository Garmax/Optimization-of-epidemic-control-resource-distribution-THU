##################################################################################
#
# PROJECT: Optimization Research of Epidemic Control Resource Distribution Network
# COURSE : General Optimization of Chemical Engineering Systems
#
# MODULE : Python Implementation for Grey Wolf Optimizer
# SOURCE : Seyedali Mirjalili, et al."Grey Wolf Optimizer." Advances in Engineering Software 69.(2014)
#
# AUTHORS: Shuaiyu Xiang, Yiming Bai.
#          Department of Chemical Engineering, Tsinghua University, P. R. China
#
##################################################################################

# --------------------------------------------------------------------------------
# Update Information
# --------------------------------------------------------------------------------
import numpy as np
import random
import math
_module_last_updated = "2020-10-06"


def initialize_pack(size=100,
                    lower_bounds=[],
                    upper_bounds=[],
                    target_function=None):
    """Initialize the original wolf pack for searching using tent map.

    Pack size by default is 100.
    """
    if len(lower_bounds) != len(upper_bounds):
        raise RuntimeError("Unmatched bound dimensions!")

    dimension = len(lower_bounds)
    half_bounds = [(j - i) / 2 for i, j in zip(lower_bounds, upper_bounds)]

    wolf_positions = np.zeros((size, dimension + 1))

    for i in range(0, dimension):
        wolf_positions[0, i] = random.uniform(lower_bounds[i], upper_bounds[i])

    wolf_positions[0, -1] = target_function(wolf_positions[0, 0:dimension])

    for i in range(1, size):
        for j in range(0, dimension):
            diff = wolf_positions[i - 1, j] - lower_bounds[j]
            if diff == half_bounds[j]:
                wolf_positions[i, j] = random.uniform(
                    lower_bounds[j], upper_bounds[j])
            else:
                new_diff = 2 * diff if diff < half_bounds[j]\
                    else 2 * (2 * half_bounds[j] - diff)
                wolf_positions[i, j] = lower_bounds[j] + new_diff

        wolf_positions[i, -1] = target_function(wolf_positions[i, 0:dimension])

    return wolf_positions


def initialize_top3(target_function=None,
                    lower_bounds=[]):
    """Initialize the top 3 wolves: Alpha, Beta & Delta."""
    return np.append(np.asarray(lower_bounds), target_function(lower_bounds))


def calculate_direction(control_factor, target, now):
    """Calculate which direction the wolf should go next."""
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    C1 = 2 * r2
    D = abs(C1 * target - now)
    A1 = control_factor * (2 * r1 - 1)
    return target - A1 * D


class Wolfpack:
    def __init__(self, size, lower_bounds, upper_bounds, target_function):
        self.pack = initialize_pack(
            size, lower_bounds, upper_bounds, target_function)
        self.size = size
        self.dimension = len(lower_bounds)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.target_function = target_function
        self.alpha = initialize_top3(target_function, lower_bounds)
        self.beta = initialize_top3(target_function, lower_bounds)
        self.delta = initialize_top3(target_function, lower_bounds)

    def update_top3(self):
        for i in range(0, self.size):
            if self.pack[i, -1] < self.alpha[-1]:
                self.alpha = np.copy(self.pack[i, :])
            elif self.pack[i, -1] < self.beta[-1]:
                self.beta = np.copy(self.pack[i, :])
            elif self.pack[i, -1] < self.delta[-1]:
                self.delta = np.copy(self.pack[i, :])

    def update_pack(self, control_factor=2):
        for i in range(0, self.size):
            for j in range(0, self.dimension):
                X1 = calculate_direction(
                    control_factor, self.alpha[j], self.pack[i, j])
                X2 = calculate_direction(
                    control_factor, self.beta[j], self.pack[i, j])
                X3 = calculate_direction(
                    control_factor, self.delta[j], self.pack[i, j])
                self.pack[i, j] = np.clip(
                    ((X1 + X2 + X3) / 3), self.lower_bounds[j], self.upper_bounds[j])
            self.pack[i, -
                      1] = self.target_function(self.pack[i, 0:self.dimension])

    def optimize(self, max_iter=100, show=True):
        count = 0
        self.update_top3()

        while count < max_iter:
            if show:
                print('Iteration {}: f(x)_min = {}.'.format(
                    count, self.alpha[-1]))
            control_factor = 2 * math.sqrt(1 - (count / max_iter))
            self.update_top3()
            self.update_pack(control_factor)
            count += 1

        print('\nIteration complete.')
        print('x = {}.'.format(self.alpha[0:self.dimension]))
        print('f(x)_min = {}'.format(self.alpha[-1]))
        return self.alpha


def optimize(target_function, lower_bounds, upper_bounds, size=100, max_iter=100, show=True):
    """Do grey wolf optimization."""
    if not isinstance(size, int) or not isinstance(max_iter, int):
        raise TypeError("Size and max iteration times must be integers.")
    if len(lower_bounds) != len(upper_bounds):
        raise RuntimeError("Unmatched lower and upper bound dimensions.")
    if len(lower_bounds) == 0:
        raise RuntimeError("No variable bounds input.")

    my_pack = Wolfpack(size, lower_bounds, upper_bounds, target_function)
    return my_pack.optimize(max_iter, show)

