#!/usr/bin/env python
import argparse
import functools
import os
import sys
import warnings
import cv2  # the import is necessary, otherwise have a bug
import keras
import keras.preprocessing.image
import tensorflow as tf
from time import time
from keras_retinanet.losses import smooth_l1, focal

# Allow relative imports when being executed as script.
# print("1", __package__)
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
#     import keras_retinanet.bin  # noqa: F401
#
#     __package__ = "keras_retinanet.bin"
# print("2", __package__)
# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.kitti import KittiGenerator
from keras_retinanet.preprocessing.open_images import OpenImagesGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.freeze_layers import freeze, freeze_by_layernum
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.callbacks.save_loss_value_callback import SaveRetinanetLossValue


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    # weights = "/home/sk49/workspace/zhoudu/keras-retinanet/snapshots/180817_resnet50_oid.epoch054-loss1.901-valloss1.903.h5"
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, model_name, backbone=None, multi_gpu=0, freeze_num=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_num    : Int, If True, disables learning for the backbone.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    # model = backbone_retinanet(num_classes)
    from keras_retinanet.models.resnet import resnet50_retinanet
    from keras_retinanet.models.resnet import resnet101_retinanet
    # model = resnet50_retinanet(num_classes)
    model = resnet101_retinanet(num_classes)
    freeze_num = 10
    model = freeze_by_layernum(model, freeze_num)
    print(model.summary())
    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(model, weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(model, weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    # compile model
    model.compile(
        loss={
            'regression': smooth_l1(),
            'classification': focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    model_path = "./snapshots/" + model_name + "_" + backbone + ".h5"
    model.save(model_path)
    print(model_path, "model saved")
    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args, model_name):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.tensorboard_dir, model_name),
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from keras_retinanet.callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    makedirs(args.snapshot_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            args.snapshot_path,
            '{model_name}_{backbone}_{dataset_type}.epoch{{epoch:03d}}-loss{{loss:.4f}}-valloss{{val_loss:.4f}}.h5'.format(
                model_name=model_name,
                backbone=args.backbone,
                dataset_type=args.dataset_type)
        ),
        verbose=1,
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=4,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    ))

    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1))
    callbacks.append(SaveRetinanetLossValue(os.path.join(args.tensorboard_dir, model_name, model_name + "_loss.csv")))

    return callbacks


def create_generators(args):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from keras_retinanet.preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            **common_args
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            **common_args
        )
    elif args.dataset_type == 'csv':
        import platform
        machine_name = platform.node()
        if machine_name == 'P100v0':
            base_dir = '/home/sk49/workspace/dataset/meter'
        elif machine_name == 'DESKTOP-3IQHBMV':
            base_dir = r'D:\DeepLearning\dataset\meter'
        else:
            base_dir = '/home/sk49/workspace/dataset/meter'
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            base_dir=base_dir,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                base_dir=base_dir,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.oid_dir,
            subset='train',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            transform_generator=transform_generator,
            **common_args
        )
        validation_generator = OpenImagesGenerator(
            args.oid_dir,
            subset='validation',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn(
            'Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet101', type=str)
    parser.add_argument('--batch_size', help='Size of the batches.', default=4, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi_gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi_gpu_force', help='Extra flag needed to enable (experimental) multi-gpu support.',
                        action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=150)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=500)
    parser.add_argument('--snapshot_path',
                        help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
                        default='./snapshots')
    parser.add_argument('--tensorboard_dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false',
                        default=False)
    parser.add_argument('--evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_true', default=True)
    parser.add_argument('--freeze_num', help='Freeze training of backbone layers.', type=int, default=0)
    parser.add_argument('--random_transform', help='Randomly transform image and annotations.', action='store_true',
                        default=True)
    parser.add_argument('--image_min_side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image_max_side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)

    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    import platform
    machine_name = platform.node()
    if machine_name == 'P100v0':
        oid_dir = '/home/sk49/workspace/dataset/open_images_dataset_v4'
    else:
        oid_dir = r'E:\datasets\open_images_dataset_v4'

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('--oid_dir', help='Path to dataset directory.',
                            default=oid_dir)
    oid_parser.add_argument('--version', help='The current dataset version is challenge2018.', default='challenge2018')
    oid_parser.add_argument('--labels_filter', help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation_cache_dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent_label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('--annotations', help='Path to CSV file containing annotations for training.',
                            default="./datas/train_groundtruth_meter_-1_retinanet_minusval.csv")
    csv_parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.',
                            default="./datas/class_mapping_meter_-1.csv")
    csv_parser.add_argument('--val-annotations',
                            help='Path to CSV file containing annotations for validation (optional).',
                            default="./datas/val_groundtruth_meter_-1_retinanet.csv")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')
    group.add_argument('--imagenet_weights',
                       help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
                       action='store_const', const=True, default=True)
    group.add_argument('--weights', help='Initialize the model with weights from a file.', )
    group.add_argument('--no_weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights',
                       action='store_const', const=False)

    return check_args(parser.parse_args(args))


def main(args=None):
    model_name = "180930"
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print("args", args)
    # args Namespace(annotation_cache_dir='.', backbone='resnet50', batch_size=1, dataset_type='oid', epochs=150,
    # evaluation=True, freeze_num=True, gpu=None, image_max_side=1333, image_min_side=800, imagenet_weights=True,
    # labels_filter=None, oid_dir='E:\\datasets\\open_images_dataset_v4', multi_gpu=0, multi_gpu_force=False,
    # parent_label=None, random_transform=True, snapshot=None, snapshot_path='./snapshots', snapshots=False, steps=2000,
    # tensorboard_dir='./logs', version='challenge2018', weights=None)

    # create object that stores backbone information
    # <keras_retinanet.models.resnet.ResNetBackbone object at 0x000001E14FD48470>
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    # check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # create the model
    print("snapshot", args.snapshot)
    preload = True
    if preload:
        print('Loading model, this may take a second...')
        weights = "./snapshots/180929_resnet50_csv.epoch014-loss0.6687-valloss0.7717.h5"
    else:
        weights = os.path.expanduser("./snapshots/resnet50_coco_best_v2.1.0.h5")
    print("weights path:", weights)

    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=train_generator.num_classes(),
        weights=weights,
        model_name=model_name,
        backbone=args.backbone,
        multi_gpu=args.multi_gpu,
        freeze_num=args.freeze_num
    )

    # print model summary
    # print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator=None,
        args=args,
        model_name=model_name
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        validation_data=validation_generator,
        validation_steps=100,
        epochs=args.epochs,
        verbose=1,
        initial_epoch=0,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
