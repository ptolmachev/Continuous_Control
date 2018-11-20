import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, qnet_params):
        super(QNetwork, self).__init__()
        self.qnet_params = qnet_params
        self.arch_params = qnet_params['arch_params']
        self.seed_as_int = qnet_params['seed']
        torch.manual_seed(self.seed_as_int)
        self.__state_dim = self.arch_params['state_and_action_dims'][0]
        self.__action_dim = self.arch_params['state_and_action_dims'][1]
        keys = list(self.arch_params['layers'].keys())
        list_of_layers = []
        prev_layer_size = self.__state_dim+self.__action_dim
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
            else:
                print("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
                break
        self.__layers = nn.ModuleList(list_of_layers)

    def forward(self, state, action):  # get action values
        """Build a network that maps state -> action values."""
        x = torch.cat((state.float(),action.float()), dim = 1)
        for i in range(len(self.__layers)):
            x = self.__layers[i](x)
        return x