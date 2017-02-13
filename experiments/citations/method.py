class PageRank(object):
    def __init__(self, alpha, iterations, tolerance):
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance
        self.current_iteration = 0

    def step(self):
        self._iterate()
        self.current_iteration += 1

    def _iterate(self):
        raise NotImplementedError()

    def get_norm(self):
        raise NotImplementedError()

    def get_vector(self):
        raise NotImplementedError()

    def has_converged(self):
        return self.current_iteration != 0 and self.get_norm() < self.tolerance

    def run(self):
        for _ in range(self.iterations):
            self.step()
            if self.has_converged():
                return
