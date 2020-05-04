import models

MODELS_DISPATCH = {
    "squeezenet" : models.SqueezeNet,
    "resnet34": models.ResNet34,
    "resnet50": models.ResNet50,
}