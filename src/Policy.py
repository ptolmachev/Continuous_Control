import torch
import torch.nn as nn
from Noise import OUNoise

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
