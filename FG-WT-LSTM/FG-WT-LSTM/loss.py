from collections import defaultdict
from typing import Dict, List, Tuple
import math

import numpy as np
import torch

from neuralhydrology.training.regularization import BaseRegularization
from neuralhydrology.utils.config import Config

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class BaseLoss(torch.nn.Module):
    """Base loss class."""

    def __init__(self,
                 cfg: Config,
                 prediction_keys: List[str],
                 ground_truth_keys: List[str],
                 additional_data: List[str] = None,
                 output_size_per_target: int = 1):
        super(BaseLoss, self).__init__()
        self._predict_last_n = _get_predict_last_n(cfg)
        self._frequencies = [f for f in self._predict_last_n.keys() if f not in cfg.no_loss_frequencies]
        self._output_size_per_target = output_size_per_target

        self._regularization_terms = []

        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        self._additional_data = []
        if additional_data is not None:
            self._additional_data = additional_data

        if cfg.target_loss_weights is None:
            weights = torch.tensor([1 / len(cfg.target_variables) for _ in range(len(cfg.target_variables))])
        else:
            if len(cfg.target_loss_weights) == len(cfg.target_variables):
                weights = torch.tensor(cfg.target_loss_weights)
            else:
                raise ValueError("Number of weights must be equal to the number of target variables")
        self._target_weights = weights

    def forward(self, prediction: Dict[str, torch.Tensor],
                data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        kwargs = {key: data[key] for key in self._additional_data}

        losses = []
        prediction_sub, ground_truth_sub = {}, {}
        for freq in self._frequencies:
            if self._predict_last_n[freq] == 0:
                continue
            freq_suffix = '' if freq == '' else f'_{freq}'

            freq_pred, freq_gt = self._subset_in_time(
                {key: prediction[f'{key}{freq_suffix}'] for key in self._prediction_keys},
                {key: data[f'{key}{freq_suffix}'] for key in self._ground_truth_keys}, self._predict_last_n[freq])

            prediction_sub.update({f'{key}{freq_suffix}': freq_pred[key] for key in freq_pred.keys()})
            ground_truth_sub.update({f'{key}{freq_suffix}': freq_gt[key] for key in freq_gt.keys()})

            for n_target, weight in enumerate(self._target_weights):
                target_pred, target_gt = self._subset_target(freq_pred, freq_gt, n_target)
                kwargs_sub = self._subset_additional_data(kwargs, n_target)
                loss = self._get_loss(target_pred, target_gt, **kwargs_sub)
                losses.append(loss * weight)

        loss = torch.sum(torch.stack(losses))
        total_loss = loss.clone()
        all_losses = defaultdict(lambda: 0)
        all_losses['loss'] = loss
        for reg_module in self._regularization_terms:
            reg_out = reg_module(prediction_sub, ground_truth_sub,
                                 {k: v for k, v in prediction.items() if k not in self._prediction_keys})
            total_loss += reg_module.weight * reg_out
            all_losses[reg_module.name] += reg_out
        all_losses['total_loss'] = total_loss
        return total_loss, all_losses

    @staticmethod
    def _subset_in_time(prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                        predict_last_n: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ground_truth_sub = {key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()}
        prediction_sub = {key: pred[:, -predict_last_n:, :] for key, pred in prediction.items()}
        return prediction_sub, ground_truth_sub

    def _subset_target(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                       n_target: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        start = n_target * self._output_size_per_target
        end = (n_target + 1) * self._output_size_per_target
        prediction_sub = {key: pred[:, :, start:end] for key, pred in prediction.items()}
        ground_truth_sub = {key: gt[:, :, n_target:n_target + 1] for key, gt in ground_truth.items()}
        return prediction_sub, ground_truth_sub

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        return additional_data

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def set_regularization_terms(self, regularization_modules: List[BaseRegularization]):
        self._regularization_terms = regularization_modules


class MaskedMSELoss(BaseLoss):
    def __init__(self, cfg: Config):
        super(MaskedMSELoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = 0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask]) ** 2)
        return loss


class MaskedRMSELoss(BaseLoss):
    def __init__(self, cfg: Config):
        super(MaskedRMSELoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = torch.sqrt(0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask]) ** 2))
        return loss


class MaskedNSELoss(BaseLoss):
    def __init__(self, cfg: Config, eps: float = 0.1):
        super(MaskedNSELoss, self).__init__(cfg,
                                            prediction_keys=['y_hat'],
                                            ground_truth_keys=['y'],
                                            additional_data=['per_basin_target_stds'])
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]
        per_basin_target_stds = kwargs['per_basin_target_stds']
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction['y_hat'])[mask]

        squared_error = (y_hat - y) ** 2
        weights = 1 / (per_basin_target_stds + self.eps) ** 2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        return {key: value[:, :, n_target:n_target + 1] for key, value in additional_data.items()}


class MaskedGMMLoss(BaseLoss):

    def __init__(self, cfg: Config, eps: float = 1e-8):
        super(MaskedGMMLoss, self).__init__(cfg,
                                            prediction_keys=['mu', 'sigma', 'pi'],
                                            ground_truth_keys=['y'],
                                            output_size_per_target=cfg.n_distributions)
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):

        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]


        m = prediction['mu'][mask]
        s = prediction['sigma'][mask]
        p = prediction['pi'][mask]

        s = torch.clamp(s, min=1e-3)
        y_expand = y.expand_as(m)

        log_prob_normal = -0.5 * math.log(2 * math.pi) - torch.log(s) - 0.5 * ((y_expand - m) / s) ** 2

        log_pi = torch.log(p + self.eps)
        log_weighted_probs = log_pi + log_prob_normal

        nll_loss = -torch.mean(torch.logsumexp(log_weighted_probs, dim=-1))

        return nll_loss

class MaskedPhysicsGMMLoss(BaseLoss):

    def __init__(self, cfg: Config, eps: float = 1e-10):
        super(MaskedPhysicsGMMLoss, self).__init__(cfg,
                                                   prediction_keys=['mu', 'sigma', 'pi'],
                                                   ground_truth_keys=['y'],
                                                   output_size_per_target=cfg.n_distributions)
        self.eps = eps

    @staticmethod
    def _gaussian_distribution(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONE_OVER_2PI_SQUARED

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        s = prediction['sigma'][mask]
        p = prediction['pi'][mask]


        s = torch.clamp(s, min=1e-3)


        y_expand = y.expand_as(m)


        log_prob_normal = -0.5 * math.log(2 * math.pi) - torch.log(s) - 0.5 * ((y_expand - m) / s) ** 2


        log_pi = torch.log(p + 1e-8)
        log_weighted_probs = log_pi + log_prob_normal


        nll_loss = -torch.mean(torch.logsumexp(log_weighted_probs, dim=-1))


        penalty_negative = torch.mean(torch.relu(-m))
        penalty_var = torch.mean(s)

        lambda_phys = 0.1
        lambda_var = 0.05

        total_loss = nll_loss + lambda_phys * penalty_negative + lambda_var * penalty_var

        return total_loss


class MaskedCMALLoss(BaseLoss):
    def __init__(self, cfg: Config, eps: float = 1e-8):
        super(MaskedCMALLoss, self).__init__(cfg,
                                             prediction_keys=['mu', 'b', 'tau', 'pi'],
                                             ground_truth_keys=['y'],
                                             output_size_per_target=cfg.n_distributions)
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]
        t = prediction['tau'][mask]
        p = prediction['pi'][mask]

        error = y - m
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b
        log_weights = torch.log(p + self.eps)

        result = torch.logsumexp(log_weights + log_like, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class MaskedUMALLoss(BaseLoss):
    def __init__(self, cfg, eps: float = 1e-5):
        super(MaskedUMALLoss, self).__init__(cfg,
                                             prediction_keys=['mu', 'b'],
                                             ground_truth_keys=['y_extended', 'tau'],
                                             output_size_per_target=2)
        self.eps = eps
        self._n_taus_count = cfg.n_taus
        self._n_taus_log = torch.as_tensor(np.log(cfg.n_taus).astype('float32'))

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y_extended']).any(1).any(1)
        y = ground_truth['y_extended'][mask]
        t = ground_truth['tau'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]

        error = y - m
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b

        original_batch_size = int(log_like.shape[0] / self._n_taus_count)
        log_like_split = torch.cat(log_like[:, :, :].split(original_batch_size, 0), 2)

        result = torch.logsumexp(log_like_split, dim=2) - self._n_taus_log
        result = -torch.mean(torch.sum(result, dim=1))
        return result


def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {'': list(predict_last_n.values())[0]}
    return predict_last_n


class MaskedLogNormalMDNLoss(BaseLoss):
    def __init__(self, cfg: Config, eps: float = 1e-5):
        n_dist = getattr(cfg, "num_mixtures", getattr(cfg, "n_distributions", 3))
        super(MaskedLogNormalMDNLoss, self).__init__(cfg,
                                                     prediction_keys=['mu', 'sigma', 'pi'],
                                                     ground_truth_keys=['y'],
                                                     output_size_per_target=n_dist)
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        y = torch.clamp(y, min=self.eps)

        m = prediction['mu'][mask]
        s = prediction['sigma'][mask]
        p = prediction['pi'][mask]

        y_expand = y.unsqueeze(-1).expand_as(m)
        log_y = torch.log(y_expand)
        var = s ** 2

        log_prob_normal = -0.5 * torch.log(2 * math.pi * var) - ((log_y - m) ** 2) / (2 * var)
        log_prob_lognormal = log_prob_normal - log_y
        log_pi = torch.log(p + 1e-8)
        log_weighted_probs = log_pi + log_prob_lognormal

        result = torch.logsumexp(log_weighted_probs, dim=-1)
        return -torch.mean(result)