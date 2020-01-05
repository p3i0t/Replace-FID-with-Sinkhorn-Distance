import torch
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import sinkhorn_pointcloud as spc


# This function is copied from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_sinkhorn_distance(samples_X, samples_Y, n=8000, epsilon=0.01, niter=100):
    # Wrap with torch tensors
    X = torch.FloatTensor(samples_X)
    Y = torch.FloatTensor(samples_Y)

    print('xxx', X.size(), Y.size())
    l1 = spc.sinkhorn_loss(X, Y, epsilon, n, niter)
    l2 = spc.sinkhorn_normalized(X, Y, epsilon, n, niter)
    return l1, l2


def get_distA_statistics(n=2000):
    mu_list = [[-1, 1], [1, 1], [-1, -1], [1, -1]]
    cov = [[0.1, 0.0], [0.0, 0.1]]

    samples_list = []
    for mu in mu_list:
        samples = np.random.multivariate_normal(mu, cov, n)
        samples_list.append(samples)
        # break

    mix_samples = np.concatenate(samples_list, axis=0)
    plot_scatter(mix_samples, 'A')

    mix_mean = np.mean(mix_samples, axis=0)
    mix_cov = np.cov(mix_samples, rowvar=False)
    return mix_mean, mix_cov, mix_samples


def get_distB_statistics(n=2000):
    sqrt_2 = np.sqrt(2)
    mu_list = [[-sqrt_2, 0], [sqrt_2, 0], [0, sqrt_2], [0, -sqrt_2]]
    cov = [[0.1, 0.0], [0.0, 0.1]]

    samples_list = []
    for mu in mu_list:
        samples = np.random.multivariate_normal(mu, cov, n)
        samples_list.append(samples)

    mix_samples = np.concatenate(samples_list, axis=0)
    plot_scatter(mix_samples, 'B')

    mix_mean = np.mean(mix_samples, axis=0)
    mix_cov = np.cov(mix_samples, rowvar=False)
    return mix_mean, mix_cov, mix_samples


def get_distC_statistics(n=2000):
    mean = [0, 0]
    cov = [[1, 0.0], [0.0, 1]]
    samples = np.random.multivariate_normal(mean, cov, n)
    plot_scatter(samples, 'C')

    mean_ = np.mean(samples, axis=0)
    cov_ = np.cov(samples, rowvar=False)
    return mean_, cov_, samples


def plot_scatter(samples, dist_name):
    x, y = samples[:, 0], samples[:, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5, edgecolors="k", c='r', linewidths=0.1)
    # plotp(x, 'b')
    # plotp(y, 'r')
    # plt.axis("off")
    plt.xlim(np.min(x) - .1, np.max(x) + .1)
    plt.ylim(np.min(y) - .1, np.max(y) + .1)
    plt.title(dist_name)
    plt.savefig('scatter_{}.png'.format(dist_name), dpi=300, pad_inches=0.1, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    n = 500
    statistics_dict = dict()
    statistics_dict['A'] = get_distA_statistics(n)
    statistics_dict['B'] = get_distB_statistics(n)
    statistics_dict['C'] = get_distC_statistics(4 * n)

    for key in statistics_dict:
        print('Distribution {}, \nmean: {}, \ncovariance: {}'.format(key, statistics_dict[key][0], statistics_dict[key][1]))

    # compute the fid between the distribution pairs
    import itertools
    for a, b in itertools.combinations(statistics_dict.keys(), 2):
        mean_a, cov_a, samples_X = statistics_dict[a]
        mean_b, cov_b, samples_Y = statistics_dict[b]

        assert samples_X.shape == samples_Y.shape
        fid = calculate_frechet_distance(mean_a, cov_a, mean_b, cov_b)

        n = samples_Y.shape[0]
        l1, l2 = cal_sinkhorn_distance(samples_X, samples_Y, n=n)
        print('======================================')
        print('Distance between {} and {}'.format(a, b))
        print('FID: {:.4f}'.format(fid))
        print('Sinkhorn loss: {:.4f}, normalized score: {:.4f}'.format(l1.item(), l2.item()))
        print('======================================')
