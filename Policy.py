import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class OUNoise:
    def __init__(self, dimension, mu=0.0, theta=0.15, sigma=0.2, seed=123):
        """Initializes the noise """
        self.dimension = dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dimension) * self.mu
        self.reset()
        np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.dimension) * self.mu

    def noise(self) -> np.ndarray:
        x = self.state
        if type(self.dimension) == tuple:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.dimension)
        elif type(self.dimension) == int:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.dimension)
        self.state = x + dx
        return self.state

class Policy(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, policy_params):
        """ arch_parameters is a dictionary like:
        {'state_and_action_dims' : (num1, num2), layers : {'Linear_1' : layer_size_1,..,'Linear_n' : layer_size_n} }
        """
        super(Policy, self).__init__()
        self.policy_params = policy_params
        self.seed_as_int = policy_params['seed']
        torch.manual_seed(self.seed_as_int)
        self.arch_params = policy_params['arch_params']
        self.__state_dim = self.arch_params['state_and_action_dims'][0]
        self.__action_dim = self.arch_params['state_and_action_dims'][1]
        self.eps = policy_params['eps']
        self.min_eps = policy_params['min_eps']
        self.eps_decay = policy_params['eps_decay']
        self.__noise_type = policy_params['noise_type']

        keys = list(self.arch_params['layers'].keys())
        if self.__noise_type == 'action':
            self.__rand_process = OUNoise((self.__action_dim,)) #, mu = 0, theta = 0.5, sigma = 0.05
        elif self.__noise_type == 'parameter':
            pass
        else:
            assert ValueError('Got an unspecified type of noise. The only available options are \'parameter\' and \'action\'')
        list_of_layers = []

        prev_layer_size = self.__state_dim
        for i in range(len(self.arch_params['layers'])):
            key = keys[i]
            layer_type = key.split('_')[0]
            if layer_type == 'Linear':
                layer_size = self.arch_params['layers'][key]
                list_of_layers.append(nn.Linear(prev_layer_size, layer_size))
                prev_layer_size = layer_size
            elif layer_type == 'LayerNorm':
                list_of_layers.append(nn.LayerNorm(prev_layer_size))
            elif layer_type == 'ReLU':
                list_of_layers.append(nn.ReLU())
            elif layer_type == 'Tanh':
                list_of_layers.append(nn.Tanh())
            else:
                print("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
                break
        self.__layers = nn.ModuleList(list_of_layers)


    def forward(self, state):  # get action values
        """Build a network that maps state -> action values."""
        y = state.float()
        for i in range(len(self.__layers)):
            y = self.__layers[i](y).float()
        y_perturbed = y + self.eps*torch.from_numpy(self.__rand_process.noise()).float()
        return y, torch.clamp(y_perturbed, min = -1.0, max = 1.0)

    # def save(self, save_to):
    #     file = {'arch_params': self.arch_params,
    #             'state_dict': self.state_dict(),
    #             'seed' : self.seed_as_int}
    #     torch.save(file, save_to)
    #
    #
    # def load(self, load_from):
    #     checkpoint = torch.load(load_from)
    #     self.__policy_params['seed'] = checkpoint['seed']
    #     self.__policy_params['arch_params'] = checkpoint['arch_params']
    #     self.__init__(self.__policy_params)
    #     self.load_state_dict(checkpoint['state_dict'])
    #     return self


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