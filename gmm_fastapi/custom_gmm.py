import numpy as np

class CustomGMM:
    def __init__(self, max_iter=100, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        # Initialize model parameters
        self.means = None
        self.covariances = None
        self.weights = None
        self.max_iter = max_iter
        self.tol = tol 
    
    @staticmethod
    def multivariate_gaussian(x, mean, cov):
        d = len(x)
        cov_inv = np.linalg.inv(cov)
        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return np.exp(exponent) / denominator

    def fit(self, X, K = 2):
        N, d = X.shape
        # Random initialization of means, covariances, and mixture coefficients
        np.random.seed(42)
        means = np.random.rand(K, d)
        covariances = np.array([np.eye(d)] * K)  # Identity matrices
        pi_k = np.ones(K) / K  # Equal priors for each cluster

        log_likelihoods = []

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities (gamma)
            gamma = np.zeros((N, K))
            for n in range(N):
                for k in range(K):
                    gamma[n, k] = pi_k[k] * self.multivariate_gaussian(X[n], means[k], covariances[k])
                gamma[n, :] /= np.sum(gamma[n, :])  # Normalize responsibilities

            # M-step: Update parameters based on the responsibilities
            N_k = np.sum(gamma, axis=0)

            for k in range(K):
                # Update means
                means[k] = np.sum(gamma[:, k].reshape(-1, 1) * X, axis=0) / N_k[k]

                # Update covariances
                covariances[k] = np.zeros((d, d))
                for n in range(N):
                    diff = X[n] - means[k]
                    covariances[k] += gamma[n, k] * np.outer(diff, diff)
                covariances[k] /= N_k[k]

                # Update mixture coefficients
                pi_k[k] = N_k[k] / N

            # Compute log likelihood
            log_likelihood = 0
            for n in range(N):
                temp = 0
                for k in range(K):
                    temp += pi_k[k] * self.multivariate_gaussian(X[n], means[k], covariances[k])
                log_likelihood += np.log(temp)
            log_likelihoods.append(log_likelihood)

            # Check for convergence
            if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

        self.means = means
        self.covariances = covariances 
        self.weights = pi_k
        self.log_likelihoods = log_likelihoods
        return means, covariances, pi_k, gamma, log_likelihoods

    def predict_proba(self, X_point):
        K = len(self.means)  # Number of clusters
        probs = np.zeros(K)

        # Compute the numerator for each cluster's responsibility
        for k in range(K):
            probs[k] = self.weights[k] * self.multivariate_gaussian(X_point, self.means[k], self.covariances[k])

        # Normalize to get probabilities
        probs /= np.sum(probs)
        return probs