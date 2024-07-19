import definitions.networks.modules as modules
import definitions.networks.mixers as mixers
import definitions.networks.resnets as resnets
import torch


def mnist_classical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        torch.nn.Linear(input, middle),
        torch.nn.ReLU(),
        torch.nn.Linear(middle, middle),
        torch.nn.ReLU(),
        torch.nn.Linear(middle, middle),
        torch.nn.ReLU(),
        torch.nn.Linear(middle, 10),
    )


def mnist_solid_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.SolidLinear(input, middle),
        modules.SolidLinear(middle, middle),
        modules.SolidLinear(middle, middle),
        modules.SolidLinear(middle, middle),
        modules.SolidLinear(middle, 10),
    )


def mnist_deep_tropical_net(input, middle, is_weight_tracking=False):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(input, middle, is_tracking=is_weight_tracking),
        modules.Max_B_Plus(middle, middle, is_tracking=is_weight_tracking),
        modules.MinPlus(middle, middle, is_tracking=is_weight_tracking),
        modules.Max_B_Plus(middle, middle, is_tracking=is_weight_tracking),
        modules.MinPlus(middle, middle, is_tracking=is_weight_tracking),
        modules.Max_B_Plus(middle, 10, is_tracking=is_weight_tracking),
    )


# def mnist_progressive_deep_tropical_net(input, middle):
#     return torch.nn.Sequential(
#         torch.nn.Flatten(-3),
#         modules.MinPlus(input, middle, factor=4),  # 4
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=0.5),  # 2
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=0.5),  # 1
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=0.5),  # 0.5
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=0.5),  # 0.25
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=1),  # 0.25
#         modules.Max_B_Plus(middle, middle),
#     )


# def mnist_regressive_deep_tropical_net(input, middle):
#     return torch.nn.Sequential(
#         torch.nn.Flatten(-3),
#         modules.MinPlus(input, middle, factor=0.25),
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=2),  # 0.5
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=2),  # 1.0
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=2),  # 2.0
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=2),  # 4.0
#         modules.Max_B_Plus(middle, middle),
#         torch.nn.Tanh(),
#         modules.MinPlus(middle, middle, factor=1),
#         modules.Max_B_Plus(middle, 10),
#     )


def tropical_resent(input):
    return torch.nn.Sequential(torch.nn.Flatten(-3), resnets.TropicalResNet(input))


def mnist_sum_tanh_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.SumTanh(input, middle),
        modules.SumTanh(middle, 10),
    )


# def mnist_mutli_encoder(input, middle):
#     return torch.nn.Sequential(
#         torch.nn.Flatten(-3),
#         modules.MinPlus(input, middle),
#         modules.Max_B_Plus(middle, middle),
#         modules.MinPlus(middle, int(middle * 0.5)),
#         modules.Max_B_Plus(int(middle * 0.5), int(middle * 0.25)),
#         modules.MinPlus(int(middle * 0.25), int(middle * 0.25)),
#         modules.Max_B_Plus(int(middle * 0.25), 10),
#     )


# def mnist_mutli_deep_encoder(input, middle):
#     return torch.nn.Sequential(
#         torch.nn.Flatten(-3),
#         modules.MinPlus(input, middle),
#         modules.Max_B_Plus(middle, middle),
#         modules.MinPlus(middle, middle),
#         modules.Max_B_Plus(middle, int(middle / 2)),
#         modules.MinPlus(int(middle / 2), int(middle / 2)),
#         modules.Max_B_Plus(int(middle / 2), int(middle / 4)),
#         modules.MinPlus(int(middle / 4), int(middle / 4)),
#         modules.Max_B_Plus(int(middle / 4), int(middle / 8)),
#         modules.MinPlus(int(middle / 8), int(middle / 8)),
#         modules.Max_B_Plus(int(middle / 8), int(middle / 16)),
#         modules.MinPlus(int(middle / 16), int(middle / 16)),
#         modules.Max_B_Plus(int(middle / 16), int(middle / 32)),
#         modules.MinPlus(int(middle / 32), int(middle / 32)),
#         modules.Max_B_Plus(int(middle / 32), int(middle / 64)),
#         modules.MinPlus(int(middle / 64), int(middle / 64)),
#         modules.Max_B_Plus(int(middle / 64), 10),
#     )


def mnist_leaky_deep_tropical_net(input, middle, is_weight_tracking=False):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.LeakyMinPlus(input, middle, is_tracking=is_weight_tracking),
        modules.LeakyMax_B_Plus(middle, middle, is_tracking=is_weight_tracking),
        modules.LeakyMinPlus(middle, middle, is_tracking=is_weight_tracking),
        modules.LeakyMax_B_Plus(middle, middle, is_tracking=is_weight_tracking),
        modules.LeakyMinPlus(middle, middle, is_tracking=is_weight_tracking),
        modules.LeakyMax_B_Plus(middle, 10, is_tracking=is_weight_tracking),
    )


def mnist_single_tropical_net(input, middle, is_weight_tracking=True, init_std=None):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(
            input, middle, is_tracking=is_weight_tracking, init_std=init_std
        ),
        modules.Max_B_Plus(
            middle, 10, is_tracking=is_weight_tracking, init_std=init_std
        ),
    )


def synthetic_single_tropical_net(input, middle, is_weight_tracking=True, init_std=None):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(
            input, middle, is_tracking=is_weight_tracking, init_std=init_std
        ),
        modules.Max_B_Plus(
            middle, 2, is_tracking=is_weight_tracking, init_std=init_std
        ),
    )


def mnist_single_scaled_tropical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.ScaledMinPlus(input, middle),
        modules.ScaledMaxPlus(middle, 10),
    )


def mnist_deep_scaled_tropical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.ScaledMinPlus(input, middle),
        modules.ScaledMaxPlus(middle, middle),
        modules.ScaledMinPlus(middle, middle),
        modules.ScaledMaxPlus(middle, middle),
        modules.ScaledMinPlus(middle, middle),
        modules.ScaledMaxPlus(middle, 10),
    )


def mnist_tropical_relu_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(input, middle),
        modules.Max_B_Plus(middle, middle),
        modules.ScaledMinPlus(middle, middle),
        modules.ScaledMaxPlus(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 10),
    )


def mnist_multi_tropical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(input, middle, is_multi=True),
        modules.Max_B_Plus(middle, 10, is_multi=True),
    )

def mnist_multi_tropical_net_tanh(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(input, middle, is_multi=True),
        modules.Max_B_Plus(middle, 10, is_multi=True),
    )


def mnist_multi_deep_tropical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinPlus(input, middle, is_multi=True),
        modules.Max_B_Plus(middle, middle, is_multi=True),
        torch.nn.Tanh(),
        modules.MinPlus(middle, middle, is_multi=True),
        modules.Max_B_Plus(middle, 10, is_multi=True),
    )


def mnist_leaky_single_tropical_net(
    input,
    middle,
    leaky_factor,
    is_random=False,
    is_weight_tracking=False,
):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.LeakyMinPlus(
            input,
            middle,
            is_tracking=is_weight_tracking,
            leaky_factor=leaky_factor,
            is_random=is_random,
        ),
        modules.LeakyMax_B_Plus(
            middle,
            10,
            is_tracking=is_weight_tracking,
            leaky_factor=leaky_factor,
            is_random=is_random,
        ),
    )


def cifar_tropical_mixer_net(middle):
    return mixers.MixerLike(
        image_size=(32, 64),
        patch_size=4,
        channels=3,
        dim=middle,
        depth=1,
        num_classes=10,
        linear=lambda in_features, out_features: torch.nn.Sequential(
            modules.MinPlus(in_features, out_features),
            modules.Max_B_Plus(out_features, out_features),
        ),
    )


def cifar_multi_mixer_net(middle):
    return mixers.MixerLike(
        image_size=(32, 64),
        patch_size=4,
        channels=3,
        dim=middle,
        depth=1,
        num_classes=10,
        linear=lambda in_features, out_features: torch.nn.Sequential(
            modules.MinPlus(in_features, out_features, is_multi=True),
            modules.Max_B_Plus(out_features, out_features, is_multi=True),
        ),
    )


def cifar_mixer_net(middle):
    return mixers.MixerLike(
        image_size=(32, 64),
        patch_size=4,
        channels=3,
        dim=middle,
        depth=1,
        num_classes=10,
        linear=modules.MinMaxSigmoid,
    )


def mnist_mixer_net(middle):
    return mixers.MixerLike(
        image_size=(28, 56),
        patch_size=4,
        channels=1,
        dim=middle,
        depth=1,
        num_classes=10,
        linear=modules.MinMaxSigmoid,
    )


def mnist_pure_sigmoid_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinMaxSigmoid(input, middle),
    )


def mnist_pure_deep_sigmoid_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinMaxSigmoid(input, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, 10),
    )


def mnist_sigmoid_net(input, middle, std_init=None):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinMaxSigmoid(input, middle, init_std=std_init),
        modules.MinPlus(middle, middle, init_std=std_init),
        modules.Max_B_Plus(middle, 10, init_std=std_init),
    )


def mnist_deep_sigmoid_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinMaxSigmoid(input, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 10),
    )


def mnist_deep_mixed_sigmoid_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Flatten(-3),
        modules.MinMaxSigmoid(input, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 10),
    )


def classical_net(input, middle):
    return torch.nn.Sequential(
        torch.nn.Linear(input, middle),
        torch.nn.Sigmoid(),
        torch.nn.Linear(middle, 1),
    )


def minmaxsigmoid(input, middle):
    return torch.nn.Sequential(
        modules.MinMaxSigmoid(input, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 1),
    )


def deep_minmaxsigmoid(input, middle):
    return torch.nn.Sequential(
        modules.MinMaxSigmoid(input, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinMaxSigmoid(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 1),
    )


def solid_net(input, middle):
    return torch.nn.Sequential(modules.Solid(input, middle), modules.Solid(input, 1))


def tropical_net_0_pm(input, middle):
    return torch.nn.Sequential(modules.MinPlus(input, middle), modules.Max_0_Plus())


def tropical_net_B_pm(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle), modules.Max_B_Plus(middle, 1)
    )


def multi_tropical_net(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle, is_multi=True),
        modules.Max_B_Plus(middle, middle, is_multi=True),
        modules.MinPlus(middle, middle, is_multi=True),
        modules.Max_B_Plus(middle, middle, is_multi=True),
        modules.MinPlus(middle, middle, is_multi=True),
        modules.Max_B_Plus(middle, 1, is_multi=True),
    )


def tropical_net_B_pm_tracked(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle, is_tracking=True),
        modules.Max_B_Plus(middle, 1, is_tracking=True),
    )


def tropical_net_B_pm_dropout_tracked(input, middle, dropout_prob=1):
    return torch.nn.Sequential(
        modules.MinPlus(
            input, middle, is_dropout=True, dropout_probability_bias=dropout_prob
        ),
        modules.Max_B_Plus(
            middle, 1, is_dropout=True, dropout_probability_bias=dropout_prob
        ),
    )


def tropical_net_B_pm_optimal_norm_tracked(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle, is_normalized=True, is_tracking=True),
        modules.Max_B_Plus(middle, 1, is_normalized=True, is_tracking=True),
    )


def tropical_net_B_pm_std_tracked(input, middle, init_std=1.0):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle, init_std=init_std, is_tracking=True),
        modules.Max_B_Plus(middle, 1, init_std=init_std, is_tracking=True),
    )


def tropical_net_B_pm_const_tracked(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle, init_const=True, is_tracking=True),
        modules.Max_B_Plus(middle, 1, init_const=True, is_tracking=True),
    )


def tropical_net_B_pm_leaky(input, middle):
    return torch.nn.Sequential(
        modules.LeakyMinPlus(input, middle, is_tracking=True),
        modules.LeakyMax_B_Plus(middle, 1, is_tracking=True),
    )


def tropical_deep_net_0_pm(input, middle):
    return torch.nn.Sequential(
        modules.MinPlus(input, middle),
        modules.Max_B_Plus(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, middle),
        modules.MinPlus(middle, middle),
        modules.Max_B_Plus(middle, 1),
    )
