# PATH_TO_FIXED_POINT_FINDER = '../fixed-point-finder'
# import sys
# sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from fixed_point_finder.examples.helper.FlipFlopData import FlipFlopData
import numpy as np

class FlipFlopDataset(FlipFlopData):
    def __init__(self,n_trials,repeats=1,**kwargs):
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.repeats = repeats


    def __call__(self):
        data = self.generate_data(n_trials=self.n_trials)
        inputs,targets = data['inputs'].swapaxes(0,1), data['targets'].swapaxes(0,1)
        inputs = np.repeat(inputs,self.repeats,axis=0)
        targets = np.repeat(targets, self.repeats, axis=0)
        return inputs, targets


class FlipFlopSweepDataset(FlipFlopDataset):
    def __init__(self,sign=1,**kwargs):
        super().__init__(**kwargs)
        self.sign = sign

    def __call__(self):
        inputs = np.zeros((self.n_time,self.n_trials,self.n_bits))
        inputs[0] = -1
        inputs[self.n_time//2,:,0] = np.linspace(-1,1, self.n_trials)
        inputs = np.repeat(inputs,self.repeats,axis=0)
        targets = np.zeros_like(inputs)
        targets[:] = np.nan

        inputs = inputs * self.sign
        return inputs, targets #[...,None,None], targets[...,None,None]


if __name__ == '__main__':
    D = FlipFlopDataset(n_trials=32,repeats=5,n_time=20)
    Dsweep = FlipFlopSweepDataset(n_trials=32, repeats=5, n_time=20)
    print(
        D()[0].shape,
        Dsweep()[0].shape,
    )




