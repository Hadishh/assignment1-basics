import math


class LRScheduler:
    def __init__(self):
        pass

    def calculate(self, iter):
        pass


class ConstantScheduler(LRScheduler):

    def __init__(self, lr):
        self.lr = lr

    def calculate(self, iter):
        return self.lr


class CosineScheduler(LRScheduler):
    def lr_cosine_schedule(self, t, alpha_max, alpha_min, t_w, t_c):

        if t < t_w:
            return (t * alpha_max) / t_w

        elif t <= t_c:
            cosine = math.cos(((t - t_w) * math.pi) / (t_c - t_w))
            cosine = 0.5 * (1 + cosine)

            return alpha_min + cosine * (alpha_max - alpha_min)

        else:
            # t > t_c
            return alpha_min

    def __init__(self, alpha_max, alpha_min, t_w, t_c):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.t_w = t_w
        self.t_c = t_c

    def calculate(self, iter):
        return self.lr_cosine_schedule(
            iter, self.alpha_max, self.alpha_min, self.t_w, self.t_c
        )
