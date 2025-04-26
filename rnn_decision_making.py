# import gymnasium as gym
# import neurogym as ngym
import matplotlib.pyplot as plt
# from neurogym.utils import info, plotting
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf_utils import omegaconf_resolvers
from config_utils import instantiate
import os
from pathlib import Path
import numpy as np
from torch import nn
import torch
# print(info.all_tasks())


CONFIG_PATH = "configs"
# CONFIG_NAME = "test"
CONFIG_NAME = "main"
project_path = os.getenv("PROJECT_PATH")

# Environment
# task = 'PerceptualDecisionMaking-v0'
# kwargs = {
#     'dt': 100,
#     'sigma': 1.0,
#     'timing': {
#         'fixation': 500,
#         'stimulus': 500,
#         'delay': ('uniform', [500, 1000]),
#     }
# }
# seq_len = 100

# task = 'GoNogo-v0'
# kwargs = {
#     'dt': 100,
#     'sigma': 0.3,
#     'timing': {
#         'delay': ('uniform', [500, 1000]),
#       # 'stim_dur': ('uniform', [10000, 11000]),
#         'stimulus': ('uniform', [500, 750]),
#     }
# }
# seq_len = 100

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    return main(cfg)

def main(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    print(OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dynamics.RNN_dataset)


    # env = dataset.env
    # ob_size = env.observation_space.shape[0]
    # act_size = env.action_space.n
    # print('ob_size',ob_size, 'act_size',act_size)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg.dynamics.RNN_model.act_size = 1
    # criterion = nn.MSELoss()
    net = instantiate(cfg.dynamics.RNN_model) #Net(num_h=64,ob_size=ob_size, act_size=act_size).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = instantiate(cfg.dynamics.RNN_criterion)
    optimizer = torch.optim.Adam(
        net.parameters(),
        # lr=1e-2,
        lr=2e-3,
        weight_decay=1e-3
    )

    running_loss = 0.0
    loss_hist = []
    for i in range(200): #100 #2000
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        # labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        labels = torch.from_numpy(labels).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # loss = criterion(outputs.view(-1, act_size), labels)
        labels = labels.to(torch.float32)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_hist.append(loss.item())
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0

    print('Finished Training')

    path = Path(cfg.savepath)
    os.makedirs(path, exist_ok=True)

    plt.figure()
    plt.plot(loss_hist)
    # plt.savefig("test_plots/"+cfg.dynamics.RNN_dataset.env+"_losses.png")

    plt.savefig(path / 'RNN_loss_hist.png',dpi=300)
    plt.close()



    inp,targ = dataset()


    torch_inp = torch.from_numpy(inp).type(torch.float).to(device)
    outputs = net(torch_inp).detach().cpu().numpy()

    fig,axs = plt.subplots(2,1,sharex=True)
    ax = axs[0]
    ax.plot(inp[:,0,:])
    ax = axs[1]
    ax.plot(targ[:, 0])
    # ax.plot(np.argmax(outputs[:, 0,:],axis=-1),ls='dashed')
    ax.plot(outputs[:, 0, :], ls='dashed')
    plt.savefig(path / "RNN_task.png")
    torch.save(net.state_dict(), os.path.join(cfg.savepath,'RNNmodel.torch'))



if __name__ == '__main__':
    decorated_main()
