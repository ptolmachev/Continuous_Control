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
        y_perturbed = self.state
        if type(self.dimension) == tuple:
            dy_perturbed = self.theta * (self.mu - y_perturbed) + self.sigma * np.random.randn(*self.dimension)
        elif type(self.dimension) == int:
            dy_perturbed = self.theta * (self.mu - y_perturbed) + self.sigma * np.random.randn(self.dimension)
        self.state = y_perturbed + dy_perturbed
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

        self.layers = nn.ModuleList(list_of_layers)

        #noise
        if self.__noise_type == 'action':
            self.__rand_process = OUNoise((self.__action_dim,))

        elif self.__noise_type == 'parameter':
            self.network_params_perturbations = dict()
            for name, parameter in self.named_parameters():
                if 'weight' in name:
                    self.network_params_perturbations[name] = OUNoise(tuple(parameter.shape))
        else:
            assert ValueError('Got an unspecified type of noise. The only available options are \'parameter\' and \'action\'')

    def forward(self, state):  # get action values
        """Build a network that maps state -> action."""


        if self.__noise_type == 'action':
            y = state.float()
            for i in range(len(self.layers)):
                y = self.layers[i](y).float()

            # #explicit forward pass using parameters of the network
            # i = -1
            # for name, parameter in self.named_parameters():
            #     if 'weight' in name:
            #         if parameter.shape[1] == y.shape[0]: #if there is a single state
            #             y = parameter.matmul(y)
            #         else:                                #if there is a batch of states
            #             y = y.matmul(parameter.t())
            #     if 'bias' in name:
            #         y = y + parameter
            #         i += 2
            #     y = self.layers[i](y).float() #assuming that every secound layer is parameterless (like Relu or Tanh)

            y_perturbed = y + self.eps*torch.from_numpy(self.__rand_process.noise()).float()
            return y, torch.clamp(y_perturbed, min = -1.0, max = 1.0)


        elif self.__noise_type == 'parameter':
            y = state.float()
            y_perturbed = state.float()
            i = -1
            for name, parameter in self.named_parameters():
                if 'weight' in name:
                    if parameter.shape[1] == y.shape[0]: #if there is a single state
                        y = parameter.matmul(y)
                        n = torch.from_numpy(self.network_params_perturbations[name].noise()).float()
                        y_perturbed = (parameter + self.eps*n).matmul(y_perturbed)
                    else:                                #if there is a batch of states
                        y = y.matmul(parameter.t())
                        n = torch.from_numpy(self.network_params_perturbations[name].noise()).float()
                        y_perturbed = y_perturbed.matmul((parameter + self.eps*n).t())
                if 'bias' in name:
                    y = y + parameter
                    y_perturbed = y_perturbed + (parameter)
                    i += 2
                y = self.layers[i](y).float()
                y_perturbed = self.layers[i](y_perturbed).float()
            return y, torch.clamp(y_perturbed, min = -1.0, max = 1.0)


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