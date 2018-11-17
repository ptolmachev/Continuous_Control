import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class OUNoise:
    """Ornstein-Uhlenbeck process
    Attributes:
        action_dimension (int): Dimension of `action`
        mu (float): 0.0
        sigma (float): > 0
        state (np.ndarray): Noise
        theta (float): > 0
    Notes:
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, dimension: int, mu=0.0, theta=0.5, sigma=0.3, seed=123) -> None:
        """Initializes the noise """
        self.dimension = dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dimension) * self.mu
        self.reset()
        self.log = []
        np.random.seed(seed)

    def reset(self) -> None:
        """Resets the states(= noise) to mu
        """
        self.state = np.ones(self.dimension) * self.mu

    def noise(self) -> np.ndarray:
        """Returns a noise(= states)
        Returns:
            np.ndarray: noise, shape (n, action_dim)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Policy(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, seed, arch_params):
        """ arch_parameters is a dictionary like:
        {'state_and_action_dims' : (num1, num2), layers : {'Linear_1' : layer_size_1,..,'Linear_n' : layer_size_n} }
        """
        super(Policy, self).__init__()
        self.seed_as_int = seed
        torch.manual_seed(seed)
        self.arch_params = arch_params
        self.state_dim = arch_params['state_and_action_dims'][0]
        self.action_dim = arch_params['state_and_action_dims'][1]
        self.eps = 1.0
        self.eps_decay = 0.993
        keys = list(self.arch_params['layers'].keys())
        # self.rand_process = OUNoise(self.action_dim) #, mu = 0, theta = 0.5, sigma = 0.05
        list_of_layers = []
        list_of_processes = []

        prev_layer_size = self.state_dim
        for i in range(len(self.arch_params['layers'])):
            key = keys[i]
            layer_type = key.split('_')[0]
            # add layer consistency check here
            if layer_type == 'Linear':
                layer_size = self.arch_params['layers'][key]
                list_of_layers.append(nn.Linear(prev_layer_size, layer_size))
                # list_of_processes.append(OUNoise(layer_size,mu = 0, theta = 0.1, sigma = 0.05))
                list_of_processes.append(None)
                prev_layer_size = layer_size
            elif layer_type == 'LayerNorm':
                list_of_layers.append(nn.LayerNorm(prev_layer_size))
                list_of_processes.append(None)
            elif layer_type == 'ReLU':
                list_of_layers.append(nn.ReLU())
                list_of_processes.append(None)
            elif layer_type == 'Tanh':
                list_of_layers.append(nn.Tanh())
                list_of_processes.append(OUNoise(layer_size)) #,mu = 0, theta = 0.1, sigma = 0.05
            else:
                print("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
                break
            self.layers = nn.ModuleList(list_of_layers)
            self.layer_noises = list_of_processes

    def forward(self, state):  # get action values
        """Build a network that maps state -> action values."""
        x = state
        y = state

        for i in range(len(self.layers)):
            if self.layer_noises[i] is not None:
                n = self.eps * torch.from_numpy(self.layer_noises[i].noise()).float()
            else:
                n = 0

            x = self.layers[i](x).float() + n
            y = self.layers[i](y).float()

        return torch.clamp(y,-1.0,1.0), torch.clamp(x,-1.0,1.0)

    def save(self, save_to):
        file = {'arch_params': self.arch_params,
                'state_dict': self.state_dict(),
                'seed' : self.seed_as_int}
        torch.save(file, save_to)


    def load(self, load_from):
        checkpoint = torch.load(load_from)
        self.__init__(checkpoint['seed'], checkpoint['arch_params'])
        self.load_state_dict(checkpoint['state_dict'])
        return self


# #quick test:
# import numpy as np
# import torch
# arch_params = {'state_and_action_dims': (10, 10),
#                'layers': {'Linear_1': 32, 'ReLU_1': None,
#                           'Linear_2': 32, 'ReLU_2': None,
#                           'Linear_4': 10}}
#
# P = Policy(3, arch_params)
# rand_vec = 1000*np.random.rand(10)-500
# state = torch.tensor(rand_vec).float()
# print(P(state))
#
# P.save('test.plc')
# arch_params = {'state_and_action_dims': (10, 6),
#                'layers': {'Linear_1': 32, 'ReLU_1': None,
#                           'Linear_2': 32, 'ReLU_2': None,
#                           'Linear_4': 6}}
# P = Policy(90,arch_params) #no matter what you put there
# P.load('test.plc')
# state = torch.tensor(rand_vec).float()
# print(P(state))