"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def freeze(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    Returns:
        keras.models.Model
    """
    for layer in model.layers:
        layer.trainable = False
    return model


def freeze_by_layernum(model, layer_num=0):
    """ freeze the model layers before the `layer_num` layers.If `layer_num` is 0, will freeze the all layers

    The weights for these layers will not be updated during training.

    Returns:
        keras.models.Model
    """
    if layer_num is 0:
        layer_num = len(model.layers)
    for layer in model.layers[:layer_num]:
        layer.trainable = False
    return model