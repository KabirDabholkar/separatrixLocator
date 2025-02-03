# PATH_TO_FIXED_POINT_FINDER = '../fixed-point-finder'
# import sys
# sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from fixed-point-finder.examples.helper.FlipFlopData import FlipFlopData



class FlipFlopDataset(FlipFlopData):
    def __init__(self,n_trials,**kwargs):
        super().__init__(**kwargs)
        self.n_trials = n_trials


    def __call__(self):
        data = self.generate_data(n_trials=self.n_trials)
        return data['inputs'].swapaxes(0,1), data['targets'].swapaxes(0,1)



if __name__ == '__main__':
    D = FlipFlopDataset(n_trials=32)
    print(
        D(),
    )




