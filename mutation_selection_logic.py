import numpy as np


class Random:
    class Mutator:
        def __init__(self, name, difference_score_total=0, total_select_times=0):
            self.name = name
            self.difference_score_total = difference_score_total
            self.total_select_times = total_select_times

    def __init__(self, mutate_ops=None):
        from dffuzz_mutator import get_mutation_ops_name
        self.mutate_ops = get_mutation_ops_name()

    def choose_mutator(self, mu1=None):
        return np.random.choice(self.mutate_ops)


class Roulette:
    class Mutator:
        def __init__(self, name, difference_score_total=0, total_select_times=0):
            self.name = name
            self.difference_score_total = difference_score_total
            self.total_select_times = total_select_times

        @property
        def score(self):
            return 1.0 / (self.total_select_times + 1)

    def __init__(self, mutate_ops=None):
        if mutate_ops is None:
            from dffuzz_mutator import get_mutation_ops_name
            mutate_ops = get_mutation_ops_name()

        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None:
            # which means it's the first mutation
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            sum = 0
            for mutator in self._mutators:
                sum += mutator.score
            rand_num = np.random.rand() * sum

            for mutator in self._mutators:
                if rand_num < mutator.score:
                    return mutator.name
                else:
                    rand_num -= mutator.score


class MCMC:
    class Mutator:
        def __init__(self, name, difference_score_total=0, total_select_times=0):
            self.name = name
            self.difference_score_total = difference_score_total
            self.fidelity_case_num = 0
            self.total_select_times = total_select_times

        @property
        def score(self):
            # use property of original testcase and mutated testcase
            # regression_faults_find
            if self.total_select_times == 0:
                return 0
            else:
                rate = self.difference_score_total * self.fidelity_case_num / (
                        self.total_select_times * self.total_select_times)  # wait to be calculated
            return rate

    def __init__(self, mutate_ops=None):
        if mutate_ops is None:
            from dffuzz_mutator import get_mutation_ops_name
            mutate_ops = get_mutation_ops_name()
        self.p = 1 / len(mutate_ops)
        print(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None:
            # which means it's the first mutation
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2.name

    def sort_mutators(self):
        import random
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        return -1
