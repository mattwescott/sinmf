import numpy as np

import scipy.signal


class SENMF(object):

    def __init__(self, n_bases, window_width, n_iter, X_shape):
        self.N_timesteps, self.N_features = X_shape
        self.n_iter = n_iter
        self.window_width = window_width
        self.n_bases = n_bases

    def rand_A(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random((self.n_bases, self.N_timesteps))+2

    def rand_D(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random((self.n_bases, self.window_width, self.N_features))+2

    def mask_D_w(self, widths):
        """
        widths is a list of length n_bases,
        each element is the width of the feature

        return a mask for D that limits window width (below window_width param)
        """
        D_m = np.zeros((self.n_bases, self.window_width, self.N_features))
        for base, w in zip(range(self.n_bases), widths):
            D_m[base, -w:, :] = np.ones((w, self.N_features))
        return D_m

    def fit(self, X, A, D):
        for _ in range(self.n_iter):
            self._update_activations(A, D, X)
            self._update_dictionary(A, D, X)

        return A, D

    def fit_A(self, X, A, D):
        for _ in range(self.n_iter):
            self._update_activations(A, D, X)

        return A

    def fit_D(self, X, A, D):
        for _ in range(self.n_iter):
            self._update_dictionary(A, D, X)

        return D

    def reconstruct(self, A, D):
        X_bar = np.zeros((self.N_timesteps, self.N_features))

        for basis, activation in zip(D, A):
            X_bar += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation)).T[:self.N_timesteps]

        return X_bar

    def _update_activations(self, A, D, X):

        for t_prime in range(self.window_width):

            X_bar = self.reconstruct(A, D)
            R = X/X_bar

            U_A = np.einsum("jk,tk->jt", D[:,t_prime,:]/np.atleast_2d(D[:,t_prime,:].sum(axis=1)).T, R[t_prime:])

            A[:,:-t_prime or None] *= U_A

    def _update_dictionary(self, A, D, X):

        for t_prime in range(self.window_width):

            X_bar = self.reconstruct(A, D)
            R = X/X_bar

            U_D = np.einsum("jn,ni->ji", A[:,:-t_prime or None]/np.atleast_2d(A[:,:-t_prime or None].sum(axis=1)).T, R[t_prime:])

            D[:,t_prime,:] *= U_D

