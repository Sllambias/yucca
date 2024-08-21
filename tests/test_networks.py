def test_2D_networks_work():
    import torch
    from yucca.modules.networks.networks import MedNeXt, UNet, UNetR, UXNet, MultiResUNet
    from yucca.functional.utils.kwargs import filter_kwargs

    n_modalities = torch.randint(low=1, high=5, size=(1,))
    n_classes = torch.randint(low=1, high=5, size=(1,))
    patch_size = (32, 32)
    data = torch.randn((2, n_modalities, *patch_size))
    conv_op = torch.nn.Conv2d
    norm_op = torch.nn.InstanceNorm2d

    kwargs = {
        "input_channels": n_modalities,
        "num_classes": n_classes,
        "conv_op": conv_op,
        "norm_op": norm_op,
        "patch_size": patch_size,
    }
    networks = [MedNeXt, UNet, UNetR, UXNet, MultiResUNet]
    for network in networks:
        kw = filter_kwargs(network, kwargs)
        net = network(**kw)
        out = net(data)
        assert out is not None


def test_3D_networks_work():
    import torch
    from yucca.modules.networks.networks import MedNeXt, UNet, UNetR, UXNet, MultiResUNet
    from yucca.functional.utils.kwargs import filter_kwargs

    n_modalities = torch.randint(low=1, high=5, size=(1,))
    n_classes = torch.randint(low=1, high=5, size=(1,))
    patch_size = (32, 32, 32)
    data = torch.randn((2, n_modalities, *patch_size))
    conv_op = torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm3d

    kwargs = {
        "input_channels": n_modalities,
        "num_classes": n_classes,
        "conv_op": conv_op,
        "norm_op": norm_op,
        "patch_size": patch_size,
    }
    networks = [MedNeXt, UNet, UNetR, UXNet, MultiResUNet]
    for network in networks:
        kw = filter_kwargs(network, kwargs)
        net = network(**kw)
        out = net(data)
        assert out is not None
