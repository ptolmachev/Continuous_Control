import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, seed, arch_params):
        """ arch_parameters is a dictionary like:
        {'state_and_action_dims' : (num1, num2), layers : {'Linear_1' : layer_size_1,..,'Linear_n' : layer_size_n} }
        """
        super(QNetwork, self).__init__()
        self.seed_as_int = seed
        torch.manual_seed(seed)
        self.arch_params = arch_params
        self.state_dim = arch_params['state_and_action_dims'][0]
        self.action_dim = arch_params['state_and_action_dims'][1]
        keys = list(self.arch_params['layers'].keys())
        list_of_layers = []
        prev_layer_size = self.state_dim+self.action_dim
        for i in range(len(self.arch_params['layers'])):
            key = keys[i]
            layer_type = key.split('_')[0]
            # add layer consistency check here
            if layer_type == 'Linear':
                layer_size = self.arch_params['layers'][key]
                list_of_layers.append(nn.Linear(prev_layer_size, layer_size))
                prev_layer_size = layer_size
            elif layer_type == 'LayerNorm':
                list_of_layers.append(nn.LayerNorm(prev_layer_size))
            elif layer_type == 'ReLU':
                list_of_layers.append(nn.ReLU())
            else:
                print("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
                break
        self.layers = nn.ModuleList(list_of_layers)

    def forward(self, state, action):  # get action values
        """Build a network that maps state -> action values."""
        x = torch.cat((state.float(),action.float()), dim = 1) # check here if concat is correct!!!
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

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



# # quick test:
# import numpy as np
# import torch
#
# arch_params = {'state_and_action_dims': (10, 10),
#                'layers': {'Linear_1': 32, 'ReLU_1': None,
#                           'Linear_2': 32, 'ReLU_2': None,
#                           'Linear_3': 32, 'ReLU_3': None,
#                           'Linear_4': 10}}
#
# Q = QNetwork(3, arch_params)
#
# state = torch.tensor(np.arange(20)).float()
# print(Q(state))
#
# Q.save('test.qnet')
# arch_params = {'state_and_action_dims': (10, 8),
#                'layers': {'Linear_1': 32, 'ReLU_1': None,
#                           'Linear_2': 32, 'ReLU_2': None,
#                           'Linear_4': 8}}
# Q2 = QNetwork(90, arch_params)  # no matter what you put there
# Q2.load('test.qnet')
# state = torch.tensor(np.arange(20)).float()
# print(Q(state))