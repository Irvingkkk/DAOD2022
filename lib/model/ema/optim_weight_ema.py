from torch.optim import Optimizer


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):  # params: teacher_detection_params  src_params: student_detection_params
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self,mutual=False):
        if mutual == True:
            print('mutual learning teacher step:',mutual)
            one_minus_alpha = 1.0 - self.alpha
            for p, src_p in zip(self.params, self.src_params):
                p.data.mul_(self.alpha)
                p.data.add_(src_p.data * one_minus_alpha)
        elif mutual == False:
            print('mutual learning teacher step:',mutual)
            one_minus_alpha = 1.0 - 0
            for p, src_p in zip(self.params, self.src_params):
                p.data.mul_(0.0)
                p.data.add_(src_p.data * one_minus_alpha)

