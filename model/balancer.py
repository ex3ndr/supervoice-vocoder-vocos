from collections import defaultdict

import torch
from torch import autograd

from .utils import average_metrics


def averager(beta = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix = defaultdict(float)
    total = defaultdict(float)

    def _update(metrics, weight = 1):
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(self, weights, rescale_grads = True, total_norm = 1.,
                 ema_decay = 0.999, per_batch_item = True, epsilon = 1e-12,
                 monitor = False):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses, input):
        norms = {}
        grads = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        avg_norms = average_metrics(self.averager(norms), count)
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                grad = grads[name] * scale
            else:
                grad = self.weights[name] * grads[name]
            out_grad += grad
        input.backward(out_grad)