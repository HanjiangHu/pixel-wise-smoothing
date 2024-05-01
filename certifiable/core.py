import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, sqrt
from statsmodels.stats.proportion import proportion_confint
from transformers_ import AbstractTransformer
import matplotlib
import matplotlib.pyplot as plt


EPS = 1e-6

class SemanticSmooth(object): 
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, transformer: AbstractTransformer, diff=False, t=-1,small=False):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.transformer = transformer
        self.diff = diff
        self.t = t
        self.small = small

    def certify(self, x: torch.tensor, n0: int, maxn: int, alpha: float, batch_size: int, cAHat=None, margin=None, indx=-1, save_flag=False, save_path=None, sigma=-1) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        self.base_classifier.eval()
        if cAHat is None:
            # draw samples of f(x+ epsilon)
            counts_selection = self._sample_noise(x, n0, save_flag=save_flag, save_path=save_path, indx=indx, sigma=sigma)
            # use these samples to take a guess at the top class
            cAHat = counts_selection.argmax().item()

        nA, n = 0, 0
        pABar = 0.0
        while n < maxn:
            now_batch = min(batch_size, maxn - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch, save_flag=save_flag, save_path=save_path, indx=indx, sigma=sigma)
            n += now_batch
            # use these samples to estimate a lower bound on pA
            nA += counts_estimation[cAHat].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
            r = self.transformer.calc_radius(pABar)
            # early stop if margin_sq is specified
            if margin is not None and r >= margin: #sqrt(margin_sq):
                # print(r, margin)
                return cAHat, r - margin #sqrt(margin_sq)
        if margin is None:
            if r <= EPS:
                return SemanticSmooth.ABSTAIN, 0.0
            else:
                return cAHat, r
        else:
            # print(pABar, r, margin)
            return (SemanticSmooth.ABSTAIN if r <= EPS else cAHat), r - margin #sqrt(margin_sq)

    def predict(self, x: torch.tensor, n0: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()

        n = 0
        counts = None
        while n < n0:
            now_batch = min(batch_size, n0 - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch)#, save_flag=save_flag, save_path=save_path, indx=indx, sigma=sigma)
            n += now_batch
            if counts is None:
                counts = counts_estimation
            else:
                counts += counts_estimation
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SemanticSmooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, save_flag=False, save_path=None, indx=-1, sigma=-1) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            if not self.diff:
                batch = [x] * num #.repeat((num, 1, 1, 1))
            else:
                batch = x.repeat((num, 1, 1, 1))
                    # print(batch.shape)
            if self.t < 0: # not using diffusion denoising
                if save_flag:
                    with_noise = []
                    if sigma < 0:
                        gaussian_sampling = np.random.randint(7000, size=len(batch))
                    else:
                        gaussian_sampling = np.random.normal(loc=3500, scale=sigma, size=len(batch)) #norm.ppf(np.random.random(len(batch)), loc=3500, scale=sigma).astype(int)
                    for i in range(len(batch)):
                        # image_i = matplotlib.image.imread(saved_path + '/%03d' % i + '/%05d.png' % 3500)
                        # img = torch.as_tensor(image_i[:, :, :3], device=torch.device('cuda'))
                        # if sigma < 0:
                        #     sample = gaussian_sampling[i]
                        # else:
                        if gaussian_sampling[i] <= 6999 and gaussian_sampling[i] >= 0:
                            sample = gaussian_sampling[i]
                        elif gaussian_sampling[i] < 0:
                            sample = 0
                        else:
                            sample = 6999
                        #   else 6999
                        # sample = gaussian_sampling[i] if gaussian_sampling[i] >= 0 else 0
                        # np.random.normal(loc=3500, scale=1167, size=10)
                        image_i_j = matplotlib.image.imread(
                            save_path + '/%03d' % indx + '/%05d.png' % sample)
                        now_img = torch.as_tensor(image_i_j[:, :, :3], device=torch.device('cuda'))
                        with_noise.append(now_img)
                    batch_noised = torch.stack(with_noise)
                else:
                    batch_noised = torch.stack([torch.as_tensor(item).cuda() for item in self.transformer.process(batch)])
                batch_noised = torch.transpose(batch_noised, 2, 3)
                batch_noised = torch.transpose(batch_noised, 1, 2).type(torch.cuda.FloatTensor)
                # print(batch_noised.shape)

                predictions = self.base_classifier(batch_noised).argmax(1)
            else:

                batch = torch.transpose(batch, 2, 3)
                batch = torch.transpose(batch, 1, 2).type(torch.cuda.FloatTensor)
                if self.small:
                    batch = torch.nn.functional.interpolate(batch, 32)
                else:
                    batch = torch.nn.functional.interpolate(batch, 256)
                # batch = torch.transpose(batch, 1, 2)
                # batch = torch.transpose(batch, 2, 3).type(torch.cuda.FloatTensor)
                # # using diffusion denoising
                # batch = torch.transpose(batch, 2, 3)
                # batch = torch.transpose(batch, 1, 2).type(torch.cuda.FloatTensor)
                predictions = self.base_classifier(batch, self.t).argmax(1)
            counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            if idx < length:
                counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]



"""
    Plan to gradually deprecate the following class.
"""
class StrictRotationSmooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, sigma_b: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.sigma_b = sigma_b

    def guess_top(self, x: torch.tensor, n0: int, batch_size: int) -> int:

        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        return cAHat

    def certify(self, x: torch.tensor, cAHat: int, maxn: int, alpha: float, batch_size: int, b: float, margin: float) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        """
        self.base_classifier.eval()
        nA, n = 0, 0
        pABar = 0.0
        while n < maxn:
            now_batch = min(batch_size, maxn - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch, batch_size)
            n += now_batch
            # use these samples to estimate a lower bound on pA
            nA += counts_estimation[cAHat].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
            if pABar >= 0.5 and norm.ppf(pABar) ** 2 - b ** 2 / max(self.sigma_b, 1e-3) ** 2 >= 0.:
                radius = (self.sigma ** 2) * (norm.ppf(pABar) ** 2 - b ** 2 / max(self.sigma_b, 1e-3) ** 2)
                # radius = self.sigma * norm.ppf(pABar)
                # radius **= 2
                if radius >= margin:
                    return cAHat, radius - margin
        if pABar < 0.5 or norm.ppf(pABar) ** 2 - b ** 2 / max(self.sigma_b, 1e-3) ** 2 < 0.:
            return SemanticSmooth.ABSTAIN, 0.0
        else:
            radius = (self.sigma ** 2) * (norm.ppf(pABar) ** 2 - b ** 2 / max(self.sigma_b, 1e-3) ** 2)
            # radius = self.sigma * norm.ppf(pABar)
            # radius **= 2
            return cAHat, radius - margin

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SemanticSmooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                noise_b = torch.randn((this_batch_size), device='cuda') * self.sigma_b
                noise_b = noise_b.reshape((this_batch_size, 1, 1, 1))
                predictions = self.base_classifier(batch + noise + noise_b).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

