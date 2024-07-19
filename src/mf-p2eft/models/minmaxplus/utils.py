import torch


def to_cpu_gradients(model):
    gradients = []
    for i in range(len(model._modules)):
        if hasattr(model[i], "W"):
            gradients.append(model[i].W.grad.cpu().detach().numpy())
    return gradients


def to_weight_regularization(network):
    weights = [
        layer.get_weight_regularizer() for layer in network.modules() if getattr(layer, "get_weight_regularizer", None)
    ]
    return weights


def to_normalized_weight_regularization(network):
    weights = [
        layer.get_normalized_weight_regularizer()
        for layer in network.modules()
        if getattr(layer, "get_normalized_weight_regularizer", None)
        and layer.get_normalized_weight_regularizer() is not None
    ]
    return weights


def to_cpu_weight_activation(network):
    layers = [module for module in network.modules() if not isinstance(module, torch.nn.Sequential)]

    weights = [
        layer.get_weight_activation()
        for layer in layers
        if getattr(layer, "get_weight_activation", None) and layer.get_weight_activation() is not None
    ]
    return weights


def to_neuron_factors(network):
    layers = [module for module in network.modules() if not isinstance(module, torch.nn.Sequential)]

    factors = [
        layer.get_factors()
        for layer in layers
        if getattr(layer, "get_factors", None) and layer.get_factors() is not None
    ]
    return factors


def to_weights(network):
    layers = [
        module
        for module in network.modules()
        if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.Flatten)
    ]

    weights = [layer.W.cpu().detach().numpy() for layer in layers]
    return weights


def reset_weight_activation(network):
    layers = [module for module in network.modules() if not isinstance(module, torch.nn.Sequential)]
    weight_activation_layers = [
        layer.reset_weight_activation()
        for layer in layers
        if getattr(layer, "reset_weight_activation", None) and layer.reset_weight_activation() is not None
    ]
    for weight_activation_layer in weight_activation_layers:
        weight_activation_layer.reset_weight_activation()


def update_weights(network, labmda_, mode):
    layers = [module for module in network.modules() if not isinstance(module, torch.nn.Sequential)]
    for layer in layers:
        if getattr(layer, "update_weights", None):
            layer.update_weights(labmda_, mode)


def reset_normalized_weight(network, labmda_):
    layers = [module for module in network.modules() if not isinstance(module, torch.nn.Sequential)]
    for layer in layers:
        if getattr(layer, "reset_normalized_weight", None):
            layer.reset_normalized_weight(labmda_)
