from pytorchvideo.models.resnet import create_resnet


def ResNet50_3D(input_channels: int, num_classes: int):
    return create_resnet(input_channel=input_channels, model_depth=50, model_num_class=num_classes)
