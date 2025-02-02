from omegaconf import OmegaConf, DictConfig
from utils import setattrs
import os


def omegaconf_resolvers():
    resolvers = {
        'eval': eval,
        'ind': lambda a, i: a[i],
        'listmul': lambda l, i: [l] * i,
        'getattr': getattr,
        'setattrs': setattrs,
        'as_tuple': lambda *args: tuple(args),
        # 'relpath': lambda p: os.path.join(
        #     '/home/kabird/lfads-torch-fewshot-benchmark', p
        # )
    }
    for resolver_name, resolver_val in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name, resolver_val)
