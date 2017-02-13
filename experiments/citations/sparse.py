import numpy


def simple(matrix, alpha, delta, iterations, norm=2):
    n = matrix.shape[0]

    # To sparse
    from scipy.sparse import csr_matrix
    sparse_matrix = csr_matrix(matrix, dtype=numpy.float)

    # To stochastic
    column_sums = numpy.array(sparse_matrix.sum(0))[0, :]  # Used to be sparse_matrix.sum(1), but I sum columns.
    ri, ci = sparse_matrix.nonzero()
    sparse_matrix.data /= column_sums[ci]

    # Bool array of sink states
    sink = (column_sums == 0) / float(n)
    sink_alpha = sink * alpha

    alpha_matrix = sparse_matrix * alpha
    ones = numpy.ones(n) / float(n)

    # Account for teleportation
    teleportation = numpy.ones(n) / float(n)
    teleportation_alpha = (1 - alpha) * teleportation

    # Compute pagerank r until we converge
    ro, r = numpy.zeros(n), numpy.ones(n) / float(n)
    iteration = 0

    def compute_norm(vector):
        if norm == 2:
            return numpy.linalg.norm(vector)
        elif norm == 1:
            return sum(abs(vector))
        else:
            raise RuntimeError("Unsupported norm: {}, should be 1 or 2 [default: 2]".format(norm))

    while compute_norm(r - ro) > delta and iteration < iterations:
        ro = r.copy()
        r = alpha * sparse_matrix * ro + (alpha * ro * sink + (1 - alpha)) * ones
        iteration += 1

    # return normalized pagerank
    return r / sum(r), iteration
