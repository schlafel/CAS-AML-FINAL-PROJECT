---
generator: "Docutils 0.17.1: http://docutils.sourceforge.net/"
title: American Sign Language Recognition 0.0.1 documentation
viewport:
- width=device-width, initial-scale=1.0
- width=device-width, initial-scale=0.9, maximum-scale=0.9
---

::: {.document}
::: {.documentwrapper}
::: {.bodywrapper}
::: {.body role="main"}
::: {#welcome-to-american-sign-language-recognition-s-documentation .section}
# Welcome to American Sign Language Recognition's documentation![¶](#welcome-to-american-sign-language-recognition-s-documentation "Permalink to this headline"){.headerlink}

::: {.toctree-wrapper .compound}
[]{#document-augmentations}

::: {#module-augmentations .section}
[]{#data-augmentations}

## Data Augmentations[¶](#module-augmentations "Permalink to this headline"){.headerlink}

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[frame_dropout]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*, *[[dropout_rate]{.pre}]{.n}[[=]{.pre}]{.o}[[0.05]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#augmentations.frame_dropout "Permalink to this definition"){.headerlink}

:   Randomly drop frames from the input landmark data.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data. dropout_rate
        (float): The proportion of frames to drop (default: 0.05).

    Returns:

    :   numpy.ndarray: An array of landmarks with dropped frames.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[mirror_landmarks]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*[)]{.sig-paren}[¶](#augmentations.mirror_landmarks "Permalink to this definition"){.headerlink}

:   Invert/mirror landmark coordinates along the x-axis.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data.

    Returns:

    :   numpy.ndarray: An array of inverted landmarks.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[normalize]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*, *[[mn]{.pre}]{.n}*, *[[std]{.pre}]{.n}*[)]{.sig-paren}[¶](#augmentations.normalize "Permalink to this definition"){.headerlink}

:   Normalize the frames with a given mean and standard deviation.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data. mn (float):
        The mean value for normalization. std (float): The standard
        deviation for normalization.

    Returns:

    :   numpy.ndarray: An array of normalized landmarks.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[random_rotation]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*, *[[max_angle]{.pre}]{.n}[[=]{.pre}]{.o}[[10]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#augmentations.random_rotation "Permalink to this definition"){.headerlink}

:   Apply random rotation to landmark coordinates. (on X and Y only)

    Args:

    :   frames (numpy.ndarray): An array of landmarks data. max_angle
        (int): The maximum rotation angle in degrees (default: 10).

    Returns:

    :   numpy.ndarray: An array of landmarks with randomly rotated
        coordinates.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[random_scaling]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*, *[[scale_range]{.pre}]{.n}[[=]{.pre}]{.o}[[(0.9,]{.pre} [1.1)]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#augmentations.random_scaling "Permalink to this definition"){.headerlink}

:   Apply random scaling to landmark coordinates.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data. scale_range
        (tuple): A tuple containing the minimum and maximum scaling
        factors (default: (0.9, 1.1)).

    Returns:

    :   numpy.ndarray: An array of landmarks with randomly scaled
        coordinates.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[shift_landmarks]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*, *[[max_shift]{.pre}]{.n}[[=]{.pre}]{.o}[[0.01]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#augmentations.shift_landmarks "Permalink to this definition"){.headerlink}

:   Shift landmark coordinates randomly by a small amount.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data. max_shift
        (float): Maximum shift for the random shift (default: 0.01).

    Returns:

    :   numpy.ndarray: An array of augmented landmarks.

```{=html}
<!-- -->
```

[[augmentations.]{.pre}]{.sig-prename .descclassname}[[standardize]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[frames]{.pre}]{.n}*[)]{.sig-paren}[¶](#augmentations.standardize "Permalink to this definition"){.headerlink}

:   Standardize the frames so that they have mean 0 and standard
    deviation 1.

    Args:

    :   frames (numpy.ndarray): An array of landmarks data.

    Returns:

    :   numpy.ndarray: An array of standardized landmarks.
:::

[]{#document-callbacks}

::: {#module-callbacks .section}
[]{#training-callbacks}

## Training Callbacks[¶](#module-callbacks "Permalink to this headline"){.headerlink}

::: {#callbacks-description .section}
### Callbacks description[¶](#callbacks-description "Permalink to this headline"){.headerlink}

This module contains callback codes which may be executed during
training. These callbacks are used to dynamically adjust the dropout
rate and data augmentation probability during the training process,
which can be useful techniques to prevent overfitting and increase the
diversity of the training data, potentially improving the model's
performance.

The dropout_callback function is designed to increase the dropout rate
of the model during the training process after a certain number of
epochs. The dropout rate is a regularization technique used to prevent
overfitting during the training process. The rate of dropout is
increased every few epochs based on a specified rate until it reaches a
specified maximum limit.

The augmentation_increase_callback: function is designed to increase the
probability of data augmentation applied to the dataset during the
training process after a certain number of epochs. Data augmentation is
a technique that can generate new training samples by applying
transformations to the existing data. The probability of data
augmentation is increased every few epochs based on a specified rate
until it reaches a specified maximum limit.

[[callbacks.]{.pre}]{.sig-prename .descclassname}[[augmentation_increase_callback]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[trainer]{.pre}]{.n}*, *[[aug_increase_rate]{.pre}]{.n}[[=]{.pre}]{.o}[[1.5]{.pre}]{.default_value}*, *[[max_limit]{.pre}]{.n}[[=]{.pre}]{.o}[[0.35]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#callbacks.augmentation_increase_callback "Permalink to this definition"){.headerlink}

:   A callback function designed to increase the probability of data
    augmentation applied on the dataset during the training process.
    Data augmentation is a technique that can generate new training
    samples by applying transformations to the existing data.

    The increase in data augmentation is performed every few epochs
    based on 'DYNAMIC_AUG_INC_INTERVAL' until it reaches a specified
    maximum limit.

    Args:

    :   trainer: The object that contains the model and handles the
        training process. aug_increase_rate: The rate at which data
        augmentation probability is increased. Default is value of
        'DYNAMIC_AUG_INC_RATE' from config. max_limit: The maximum limit
        to which data augmentation probability can be increased. Default
        is value of 'DYNAMIC_AUG_MAX_THRESHOLD' from config.

    Returns:

    :   None

    Functionality:

    :   Increases the probability of data augmentation applied on the
        dataset after certain number of epochs defined by
        'DYNAMIC_AUG_INC_INTERVAL'.

    Parameters

    :   -   **trainer** -- Trainer object handling the training process

        -   **aug_increase_rate** -- Rate at which to increase the data
            augmentation probability

        -   **max_limit** -- Maximum allowable data augmentation
            probability

    Returns

    :   None

    Return type

    :   None

```{=html}
<!-- -->
```

[[callbacks.]{.pre}]{.sig-prename .descclassname}[[dropout_callback]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[trainer]{.pre}]{.n}*, *[[dropout_rate]{.pre}]{.n}[[=]{.pre}]{.o}[[1.1]{.pre}]{.default_value}*, *[[max_dropout]{.pre}]{.n}[[=]{.pre}]{.o}[[0.2]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#callbacks.dropout_callback "Permalink to this definition"){.headerlink}

:   A callback function designed to increase the dropout rate of the
    model in training after a certain number of epochs. The dropout rate
    is a regularization technique which helps in preventing overfitting
    during the training process.

    The rate of dropout is increased every few epochs based on the
    config parameter (in config.py)
    'DYNAMIC_DROP_OUT_REDUCTION_INTERVAL' until a maximum threshold
    defined by 'max_dropout'. This function is usually called after each
    epoch in the training process.

    Args:

    :   trainer: The object that contains the model and handles the
        training process. dropout_rate: The rate at which the dropout
        rate is increased. Default is value of
        'DYNAMIC_DROP_OUT_REDUCTION_RATE' from config. max_dropout: The
        maximum limit to which dropout can be increased. Default is
        value of 'DYNAMIC_DROP_OUT_MAX_THRESHOLD' from config.

    Returns:

    :   None

    Functionality:

    :   Increases the dropout rate of all nn.Dropout modules in the
        model after certain number of epochs defined by
        'DYNAMIC_DROP_OUT_REDUCTION_INTERVAL'.

    Parameters

    :   -   **trainer** -- Trainer object handling the training process

        -   **dropout_rate** -- Rate at which to increase the dropout
            rate

        -   **max_dropout** -- Maximum allowable dropout rate

    Returns

    :   None

    Return type

    :   None
:::
:::

[]{#document-config}

::: {#module-config .section}
[]{#project-configuration}

## Project Configuration[¶](#module-config "Permalink to this headline"){.headerlink}

::: {#project-configuration-description .section}
### Project configuration description[¶](#project-configuration-description "Permalink to this headline"){.headerlink}

This configuration is created allows for easy tuning of your machine
learning model's parameters and setup. The device on which the model
runs, the paths for various resources, the seed for random number
generation, hyperparameters for model training, and much more and
quickly be change and configured. This makes your setup flexible and
easy to adapt for various experiments and environments

[[config.]{.pre}]{.sig-prename .descclassname}[[BATCH_SIZE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[128]{.pre}*[¶](#config.BATCH_SIZE "Permalink to this definition"){.headerlink}

:   Training Batch Size

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[CHECKPOINT_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'checkpoints/\']{.pre}*[¶](#config.CHECKPOINT_DIR "Permalink to this definition"){.headerlink}

:   Checkpoint files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[CLEANED_FILE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'cleansed_data.marker\']{.pre}*[¶](#config.CLEANED_FILE "Permalink to this definition"){.headerlink}

:   File that marks the data cleaning stage.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[COLUMNS_TO_USE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\[\'x\',]{.pre} [\'y\'\]]{.pre}*[¶](#config.COLUMNS_TO_USE "Permalink to this definition"){.headerlink}

:   Coordinate columns from the data to use for training.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DATA_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'data/\']{.pre}*[¶](#config.DATA_DIR "Permalink to this definition"){.headerlink}

:   Data files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DEVICE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'cpu\']{.pre}*[¶](#config.DEVICE "Permalink to this definition"){.headerlink}

:   Setting the device for training, 'cuda' if a CUDA-compatible GPU is
    available, 'mps' if multiple processors are available, 'cpu' if none
    of the above.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DL_FRAMEWORK]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'pytorch\']{.pre}*[¶](#config.DL_FRAMEWORK "Permalink to this definition"){.headerlink}

:   Deep learning framework to use for training and inference. Can be
    either 'pytorch' or 'tensorflow'.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_AUG_INC_INTERVAL]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[5]{.pre}*[¶](#config.DYNAMIC_AUG_INC_INTERVAL "Permalink to this definition"){.headerlink}

:   The number of epochs to wait before increasing the probability of
    data augmentation.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_AUG_INC_RATE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[1.5]{.pre}*[¶](#config.DYNAMIC_AUG_INC_RATE "Permalink to this definition"){.headerlink}

:   The rate at which the probability of data augmentation is increased.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_AUG_MAX_THRESHOLD]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.35]{.pre}*[¶](#config.DYNAMIC_AUG_MAX_THRESHOLD "Permalink to this definition"){.headerlink}

:   The maximum limit to which the probability of data augmentation can
    be increased.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_DROP_OUT_INIT_RATE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.01]{.pre}*[¶](#config.DYNAMIC_DROP_OUT_INIT_RATE "Permalink to this definition"){.headerlink}

:   The value of initial low dropouts rate

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_DROP_OUT_MAX_THRESHOLD]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.2]{.pre}*[¶](#config.DYNAMIC_DROP_OUT_MAX_THRESHOLD "Permalink to this definition"){.headerlink}

:   The max value of dynamic dropouts

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_DROP_OUT_REDUCTION_INTERVAL]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[2]{.pre}*[¶](#config.DYNAMIC_DROP_OUT_REDUCTION_INTERVAL "Permalink to this definition"){.headerlink}

:   The epoch interval value to gradually change dropout rate

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[DYNAMIC_DROP_OUT_REDUCTION_RATE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[1.1]{.pre}*[¶](#config.DYNAMIC_DROP_OUT_REDUCTION_RATE "Permalink to this definition"){.headerlink}

:   The value to increase dropouts by

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[EARLY_STOP_METRIC]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'accuracy\']{.pre}*[¶](#config.EARLY_STOP_METRIC "Permalink to this definition"){.headerlink}

:   Which metric should be used for early stopping loss/accuracy

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[EARLY_STOP_MODE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'max\']{.pre}*[¶](#config.EARLY_STOP_MODE "Permalink to this definition"){.headerlink}

:   What is the mode? min/max

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[EARLY_STOP_PATIENCE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[10]{.pre}*[¶](#config.EARLY_STOP_PATIENCE "Permalink to this definition"){.headerlink}

:   The number of epochs to wait for improvement in the validation loss
    before stopping training

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[EARLY_STOP_TOLERENCE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.001]{.pre}*[¶](#config.EARLY_STOP_TOLERENCE "Permalink to this definition"){.headerlink}

:   The value of loss as margin to tolerate

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[EPOCHS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[50]{.pre}*[¶](#config.EPOCHS "Permalink to this definition"){.headerlink}

:   Training Number of epochs

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[FACE_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[468]{.pre}*[¶](#config.FACE_FEATURES "Permalink to this definition"){.headerlink}

:   Number of features related to the face in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[FACE_FEATURE_START]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0]{.pre}*[¶](#config.FACE_FEATURE_START "Permalink to this definition"){.headerlink}

:   Start index for face feature in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[FACE_INDICES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[]{.pre} [0,]{.pre}  [1,]{.pre}  [2,]{.pre}  [3,]{.pre}  [4,]{.pre}  [5,]{.pre}  [6,]{.pre}  [7,]{.pre}  [8,]{.pre}  [9,]{.pre} [10,]{.pre} [11,]{.pre} [12,]{.pre} [13,]{.pre} [14,]{.pre} [15,]{.pre} [16,]{.pre}        [17,]{.pre} [18,]{.pre} [19,]{.pre} [20,]{.pre} [21,]{.pre} [22,]{.pre} [23,]{.pre} [24,]{.pre} [25,]{.pre} [26,]{.pre} [27,]{.pre} [28,]{.pre} [29,]{.pre} [30,]{.pre} [31,]{.pre} [32,]{.pre} [33,]{.pre}        [34,]{.pre} [35,]{.pre} [36,]{.pre} [37,]{.pre} [38,]{.pre} [39\],]{.pre} [dtype=int64)]{.pre}*[¶](#config.FACE_INDICES "Permalink to this definition"){.headerlink}

:   Indices of face landmarks that are used from the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[FACE_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[]{.pre} [61,]{.pre} [185,]{.pre}  [40,]{.pre}  [39,]{.pre}  [37,]{.pre}   [0,]{.pre} [267,]{.pre} [269,]{.pre} [270,]{.pre} [409,]{.pre} [291,]{.pre} [146,]{.pre}  [91,]{.pre}        [181,]{.pre}  [84,]{.pre}  [17,]{.pre} [314,]{.pre} [405,]{.pre} [321,]{.pre} [375,]{.pre}  [78,]{.pre} [191,]{.pre}  [80,]{.pre}  [81,]{.pre}  [82,]{.pre}  [13,]{.pre}        [312,]{.pre} [311,]{.pre} [310,]{.pre} [415,]{.pre}  [95,]{.pre}  [88,]{.pre} [178,]{.pre}  [87,]{.pre}  [14,]{.pre} [317,]{.pre} [402,]{.pre} [318,]{.pre} [324,]{.pre}        [308\])]{.pre}*[¶](#config.FACE_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for Lips

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[HAND_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[21]{.pre}*[¶](#config.HAND_FEATURES "Permalink to this definition"){.headerlink}

:   Number of features related to the hand in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[HAND_INDICES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[40,]{.pre} [41,]{.pre} [42,]{.pre} [43,]{.pre} [44,]{.pre} [45,]{.pre} [46,]{.pre} [47,]{.pre} [48,]{.pre} [49,]{.pre} [50,]{.pre} [51,]{.pre} [52,]{.pre} [53,]{.pre} [54,]{.pre} [55,]{.pre} [56,]{.pre}        [57,]{.pre} [58,]{.pre} [59,]{.pre} [60,]{.pre} [61,]{.pre} [62,]{.pre} [63,]{.pre} [64,]{.pre} [65,]{.pre} [66,]{.pre} [67,]{.pre} [68,]{.pre} [69,]{.pre} [70,]{.pre} [71,]{.pre} [72,]{.pre} [73,]{.pre}        [74,]{.pre} [75,]{.pre} [76,]{.pre} [77,]{.pre} [78,]{.pre} [79,]{.pre} [80,]{.pre} [81\],]{.pre} [dtype=int64)]{.pre}*[¶](#config.HAND_INDICES "Permalink to this definition"){.headerlink}

:   Indices of hand landmarks that are used from the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[INPUT_SIZE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[32]{.pre}*[¶](#config.INPUT_SIZE "Permalink to this definition"){.headerlink}

:   Size of the input data for the model.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[INTEREMOLATE_MISSING]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[3]{.pre}*[¶](#config.INTEREMOLATE_MISSING "Permalink to this definition"){.headerlink}

:   Number of missing values to interpolate in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[LANDMARK_FILES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'train_landmark_files\']{.pre}*[¶](#config.LANDMARK_FILES "Permalink to this definition"){.headerlink}

:   Directory where training landmark files are stored.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[LEARNING_RATE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.001]{.pre}*[¶](#config.LEARNING_RATE "Permalink to this definition"){.headerlink}

:   Training Learning rate

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[LEFT_HAND_FEATURE_START]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[468]{.pre}*[¶](#config.LEFT_HAND_FEATURE_START "Permalink to this definition"){.headerlink}

:   Start index for left hand feature in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[LEFT_HAND_INDICES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[40,]{.pre} [41,]{.pre} [42,]{.pre} [43,]{.pre} [44,]{.pre} [45,]{.pre} [46,]{.pre} [47,]{.pre} [48,]{.pre} [49,]{.pre} [50,]{.pre} [51,]{.pre} [52,]{.pre} [53,]{.pre} [54,]{.pre} [55,]{.pre} [56,]{.pre}        [57,]{.pre} [58,]{.pre} [59,]{.pre} [60\],]{.pre} [dtype=int64)]{.pre}*[¶](#config.LEFT_HAND_INDICES "Permalink to this definition"){.headerlink}

:   Indices of left hand landmarks that are used from the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MAP_JSON_FILE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'sign_to_prediction_index_map.json\']{.pre}*[¶](#config.MAP_JSON_FILE "Permalink to this definition"){.headerlink}

:   JSON file that maps sign to prediction index.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MARKER_FILE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'preprocessed_data.marker\']{.pre}*[¶](#config.MARKER_FILE "Permalink to this definition"){.headerlink}

:   File that marks the preprocessing stage.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MAX_SEQUENCES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[32]{.pre}*[¶](#config.MAX_SEQUENCES "Permalink to this definition"){.headerlink}

:   Maximum number of sequences in the input data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MIN_SEQUEENCES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[8.0]{.pre}*[¶](#config.MIN_SEQUEENCES "Permalink to this definition"){.headerlink}

:   Minimum number of sequences in the input data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MODELNAME]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'YetAnotherTransformerClassifier\']{.pre}*[¶](#config.MODELNAME "Permalink to this definition"){.headerlink}

:   Name of the model to be used for training.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[MODEL_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'models/\']{.pre}*[¶](#config.MODEL_DIR "Permalink to this definition"){.headerlink}

:   Model files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[N_CLASSES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[250]{.pre}*[¶](#config.N_CLASSES "Permalink to this definition"){.headerlink}

:   Number of classes

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[N_DIMS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[2]{.pre}*[¶](#config.N_DIMS "Permalink to this definition"){.headerlink}

:   Number of dimensions used in training

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[N_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[96]{.pre}*[¶](#config.N_LANDMARKS "Permalink to this definition"){.headerlink}

:   Total number of used landmarks

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[OUT_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'out/\']{.pre}*[¶](#config.OUT_DIR "Permalink to this definition"){.headerlink}

:   Output files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[POSE_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[33]{.pre}*[¶](#config.POSE_FEATURES "Permalink to this definition"){.headerlink}

:   Number of features related to the pose in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[POSE_FEATURE_START]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[489]{.pre}*[¶](#config.POSE_FEATURE_START "Permalink to this definition"){.headerlink}

:   Start index for pose feature in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[POSE_INDICES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[82,]{.pre} [83,]{.pre} [84,]{.pre} [85,]{.pre} [86,]{.pre} [87,]{.pre} [88,]{.pre} [89,]{.pre} [90,]{.pre} [91,]{.pre} [92,]{.pre} [93,]{.pre} [94,]{.pre} [95\],]{.pre}       [dtype=int64)]{.pre}*[¶](#config.POSE_INDICES "Permalink to this definition"){.headerlink}

:   Indices of pose landmarks that are used from the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[PROCESSED_DATA_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'data/processed/\']{.pre}*[¶](#config.PROCESSED_DATA_DIR "Permalink to this definition"){.headerlink}

:   Processed Data files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[RAW_DATA_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'data/raw/\']{.pre}*[¶](#config.RAW_DATA_DIR "Permalink to this definition"){.headerlink}

:   Raw Data files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[RIGHT_HAND_FEATURE_START]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[522]{.pre}*[¶](#config.RIGHT_HAND_FEATURE_START "Permalink to this definition"){.headerlink}

:   Start index for right hand feature in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[RIGHT_HAND_INDICES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[61,]{.pre} [62,]{.pre} [63,]{.pre} [64,]{.pre} [65,]{.pre} [66,]{.pre} [67,]{.pre} [68,]{.pre} [69,]{.pre} [70,]{.pre} [71,]{.pre} [72,]{.pre} [73,]{.pre} [74,]{.pre} [75,]{.pre} [76,]{.pre} [77,]{.pre}        [78,]{.pre} [79,]{.pre} [80,]{.pre} [81\],]{.pre} [dtype=int64)]{.pre}*[¶](#config.RIGHT_HAND_INDICES "Permalink to this definition"){.headerlink}

:   Indices of right hand landmarks that are used from the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[ROOT_PATH]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'C:\\\\Users\\\\tgdimas1\\\\git\\\\CAS-AML-FINAL-PROJECT\\\\src\\\\../\']{.pre}*[¶](#config.ROOT_PATH "Permalink to this definition"){.headerlink}

:   Root directory

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[ROWS_PER_FRAME]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[543]{.pre}*[¶](#config.ROWS_PER_FRAME "Permalink to this definition"){.headerlink}

:   Number of rows per frame in the data.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[RUNS_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'runs/\']{.pre}*[¶](#config.RUNS_DIR "Permalink to this definition"){.headerlink}

:   Run files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[SEED]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0]{.pre}*[¶](#config.SEED "Permalink to this definition"){.headerlink}

:   Set Random Seed

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[SKIP_CONSECUTIVE_ZEROS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[4]{.pre}*[¶](#config.SKIP_CONSECUTIVE_ZEROS "Permalink to this definition"){.headerlink}

:   Skip data if there are this many consecutive zeros.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[SRC_DIR]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'src/\']{.pre}*[¶](#config.SRC_DIR "Permalink to this definition"){.headerlink}

:   Source files Directory path

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[TEST_SIZE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.05]{.pre}*[¶](#config.TEST_SIZE "Permalink to this definition"){.headerlink}

:   Testing Test set size

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[TRAIN_CSV_FILE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[\'train.csv\']{.pre}*[¶](#config.TRAIN_CSV_FILE "Permalink to this definition"){.headerlink}

:   CSV file name that contains the training dataset.

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[TRAIN_SIZE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.9]{.pre}*[¶](#config.TRAIN_SIZE "Permalink to this definition"){.headerlink}

:   Training Train set split size

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[TUNE_HP]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[True]{.pre}*[¶](#config.TUNE_HP "Permalink to this definition"){.headerlink}

:   Tune hyperparameters

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USED_FACE_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[40]{.pre}*[¶](#config.USED_FACE_FEATURES "Permalink to this definition"){.headerlink}

:   Count of facial features used

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USED_HAND_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[21]{.pre}*[¶](#config.USED_HAND_FEATURES "Permalink to this definition"){.headerlink}

:   Count of hands features used (single hand only)

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USED_POSE_FEATURES]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[14]{.pre}*[¶](#config.USED_POSE_FEATURES "Permalink to this definition"){.headerlink}

:   Count of body/pose features used

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_ALL_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[]{.pre} [61,]{.pre} [185,]{.pre}  [40,]{.pre}  [39,]{.pre}  [37,]{.pre}   [0,]{.pre} [267,]{.pre} [269,]{.pre} [270,]{.pre} [409,]{.pre} [291,]{.pre} [146,]{.pre}  [91,]{.pre}        [181,]{.pre}  [84,]{.pre}  [17,]{.pre} [314,]{.pre} [405,]{.pre} [321,]{.pre} [375,]{.pre}  [78,]{.pre} [191,]{.pre}  [80,]{.pre}  [81,]{.pre}  [82,]{.pre}  [13,]{.pre}        [312,]{.pre} [311,]{.pre} [310,]{.pre} [415,]{.pre}  [95,]{.pre}  [88,]{.pre} [178,]{.pre}  [87,]{.pre}  [14,]{.pre} [317,]{.pre} [402,]{.pre} [318,]{.pre} [324,]{.pre}        [308,]{.pre} [468,]{.pre} [469,]{.pre} [470,]{.pre} [471,]{.pre} [472,]{.pre} [473,]{.pre} [474,]{.pre} [475,]{.pre} [476,]{.pre} [477,]{.pre} [478,]{.pre} [479,]{.pre}        [480,]{.pre} [481,]{.pre} [482,]{.pre} [483,]{.pre} [484,]{.pre} [485,]{.pre} [486,]{.pre} [487,]{.pre} [488,]{.pre} [522,]{.pre} [523,]{.pre} [524,]{.pre} [525,]{.pre}        [526,]{.pre} [527,]{.pre} [528,]{.pre} [529,]{.pre} [530,]{.pre} [531,]{.pre} [532,]{.pre} [533,]{.pre} [534,]{.pre} [535,]{.pre} [536,]{.pre} [537,]{.pre} [538,]{.pre}        [539,]{.pre} [540,]{.pre} [541,]{.pre} [542,]{.pre} [500,]{.pre} [501,]{.pre} [502,]{.pre} [503,]{.pre} [504,]{.pre} [505,]{.pre} [506,]{.pre} [507,]{.pre} [508,]{.pre}        [509,]{.pre} [510,]{.pre} [511,]{.pre} [512,]{.pre} [513\])]{.pre}*[¶](#config.USEFUL_ALL_LANDMARKS "Permalink to this definition"){.headerlink}

:   All Landmarks

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_FACE_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[]{.pre} [61,]{.pre} [185,]{.pre}  [40,]{.pre}  [39,]{.pre}  [37,]{.pre}   [0,]{.pre} [267,]{.pre} [269,]{.pre} [270,]{.pre} [409,]{.pre} [291,]{.pre} [146,]{.pre}  [91,]{.pre}        [181,]{.pre}  [84,]{.pre}  [17,]{.pre} [314,]{.pre} [405,]{.pre} [321,]{.pre} [375,]{.pre}  [78,]{.pre} [191,]{.pre}  [80,]{.pre}  [81,]{.pre}  [82,]{.pre}  [13,]{.pre}        [312,]{.pre} [311,]{.pre} [310,]{.pre} [415,]{.pre}  [95,]{.pre}  [88,]{.pre} [178,]{.pre}  [87,]{.pre}  [14,]{.pre} [317,]{.pre} [402,]{.pre} [318,]{.pre} [324,]{.pre}        [308\])]{.pre}*[¶](#config.USEFUL_FACE_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for face

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_HAND_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[468,]{.pre} [469,]{.pre} [470,]{.pre} [471,]{.pre} [472,]{.pre} [473,]{.pre} [474,]{.pre} [475,]{.pre} [476,]{.pre} [477,]{.pre} [478,]{.pre} [479,]{.pre} [480,]{.pre}        [481,]{.pre} [482,]{.pre} [483,]{.pre} [484,]{.pre} [485,]{.pre} [486,]{.pre} [487,]{.pre} [488,]{.pre} [522,]{.pre} [523,]{.pre} [524,]{.pre} [525,]{.pre} [526,]{.pre}        [527,]{.pre} [528,]{.pre} [529,]{.pre} [530,]{.pre} [531,]{.pre} [532,]{.pre} [533,]{.pre} [534,]{.pre} [535,]{.pre} [536,]{.pre} [537,]{.pre} [538,]{.pre} [539,]{.pre}        [540,]{.pre} [541,]{.pre} [542\])]{.pre}*[¶](#config.USEFUL_HAND_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for both hands

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_LEFT_HAND_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[468,]{.pre} [469,]{.pre} [470,]{.pre} [471,]{.pre} [472,]{.pre} [473,]{.pre} [474,]{.pre} [475,]{.pre} [476,]{.pre} [477,]{.pre} [478,]{.pre} [479,]{.pre} [480,]{.pre}        [481,]{.pre} [482,]{.pre} [483,]{.pre} [484,]{.pre} [485,]{.pre} [486,]{.pre} [487,]{.pre} [488\])]{.pre}*[¶](#config.USEFUL_LEFT_HAND_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for left hand

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_POSE_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[500,]{.pre} [501,]{.pre} [502,]{.pre} [503,]{.pre} [504,]{.pre} [505,]{.pre} [506,]{.pre} [507,]{.pre} [508,]{.pre} [509,]{.pre} [510,]{.pre} [511,]{.pre} [512,]{.pre}        [513\])]{.pre}*[¶](#config.USEFUL_POSE_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for pose

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[USEFUL_RIGHT_HAND_LANDMARKS]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[array(\[522,]{.pre} [523,]{.pre} [524,]{.pre} [525,]{.pre} [526,]{.pre} [527,]{.pre} [528,]{.pre} [529,]{.pre} [530,]{.pre} [531,]{.pre} [532,]{.pre} [533,]{.pre} [534,]{.pre}        [535,]{.pre} [536,]{.pre} [537,]{.pre} [538,]{.pre} [539,]{.pre} [540,]{.pre} [541,]{.pre} [542\])]{.pre}*[¶](#config.USEFUL_RIGHT_HAND_LANDMARKS "Permalink to this definition"){.headerlink}

:   Landmarks for right hand

```{=html}
<!-- -->
```

[[config.]{.pre}]{.sig-prename .descclassname}[[VALID_SIZE]{.pre}]{.sig-name .descname}*[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[0.05]{.pre}*[¶](#config.VALID_SIZE "Permalink to this definition"){.headerlink}

:   Training Validation set size
:::
:::

[]{#document-data_utils}

::: {#module-data.data_utils .section}
[]{#data-utilities}

## Data Utilities[¶](#module-data.data_utils "Permalink to this headline"){.headerlink}

::: {#data-processing-utils-description .section}
### Data processing Utils description[¶](#data-processing-utils-description "Permalink to this headline"){.headerlink}

This module handles the loading and preprocessing of data. It is
specifically tailored for loading ASL sign language dataset where the
raw data includes information about the position of hands, face, and
body over time.

ASL stands for American Sign Language, which is a natural language used
by individuals who are deaf or hard of hearing to communicate through
hand gestures and facial expressions.

The dataset consists of sequences of frames, where each frame contains
multiple "landmarks". Each of these landmarks has multiple features,
such as coordinates. The landmarks may represent various aspects of
human body, such as facial features, hand positions, and body pose.

This module is used to process the raw data, to create a uniform dataset
where all sequences are of the same length and all missing values have
been handled in a way that maintains the integrity of the data. This
involves steps like detecting and removing empty frames, selecting
specific landmarks, resizing sequences and handling NaN values.

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[calculate_avg_landmark_positions]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.calculate_avg_landmark_positions "Permalink to this definition"){.headerlink}

:   Calculate the average landmark positions for left-hand, right-hand,
    and face landmarks for each sign in the dataset. The purpose of this
    function is to compute the average positions of landmarks for
    left-hand, right-hand, and face for each sign in the training
    dataset.

    Returns: List : Containing a dictionary with average x/y positions
    with keys - 'left_hand' - 'right_hand' - 'face'

    Functionality: - The function takes an ASLDataset object as an
    input, which contains the training data. - It calculates the average
    landmark positions for left-hand, right-hand, and face landmarks for
    each sign in the dataset. - The function returns a list containing a
    dictionary with average x/y positions with keys 'left_hand',
    'right_hand', and 'face' for each sign.

    Parameters

    :   **dataset**
        ([*ASL_DATASET*](index.html#data.dataset.ASL_DATASET "data.dataset.ASL_DATASET"){.reference
        .internal}) -- The ASL dataset object containing the training
        data.

    Returns

    :   A list containing a dictionary with average x/y positions with
        keys 'left_hand', 'right_hand', and 'face'

    for each sign. :rtype: List\[Dict\[str, np.ndarray\]\]

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[calculate_landmark_length_stats]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data.data_utils.calculate_landmark_length_stats "Permalink to this definition"){.headerlink}

:   Calculate statistics of landmark lengths for each sign type.

    Returns: dict: A dictionary of landmark lengths for each sign type
    containing: - minimum - maximum - mean - median - standard deviation

    Functionality: - The function reads the CSV file. - It groups the
    DataFrame by sign. - An empty dictionary is created to store average
    landmarks for each sign type. - The function loops through each
    unique sign and its corresponding rows in the grouped DataFrame. -
    For each sign, it initializes a list to store the length of
    landmarks for each example of the current sign. - It loops through
    each row of the current sign type, loads the data, and adds the
    length of landmarks of the current example to the list of current
    sign data. - The function calculates the minimum, maximum, mean,
    standard deviation, and median of the landmarks for the current sign
    and updates the dictionary. - The resulting dictionary containing
    average landmarks for each sign type is returned.

    Returns

    :   A dictionary of landmark lengths for each sign type containing
        minimum, maximum, mean, median & standard

    deviation :rtype: dict

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[create_data_loaders]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[asl_dataset]{.pre}]{.n}*, *[[train_size]{.pre}]{.n}[[=]{.pre}]{.o}[[0.9]{.pre}]{.default_value}*, *[[valid_size]{.pre}]{.n}[[=]{.pre}]{.o}[[0.05]{.pre}]{.default_value}*, *[[test_size]{.pre}]{.n}[[=]{.pre}]{.o}[[0.05]{.pre}]{.default_value}*, *[[batch_size]{.pre}]{.n}[[=]{.pre}]{.o}[[128]{.pre}]{.default_value}*, *[[random_state]{.pre}]{.n}[[=]{.pre}]{.o}[[0]{.pre}]{.default_value}*, *[[dl_framework]{.pre}]{.n}[[=]{.pre}]{.o}[[\'pytorch\']{.pre}]{.default_value}*, *[[num_workers]{.pre}]{.n}[[=]{.pre}]{.o}[[8]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#data.data_utils.create_data_loaders "Permalink to this definition"){.headerlink}

:   Split the ASL dataset into training, validation, and testing sets
    and create data loaders for each set.

    Args: asl_dataset (ASLDataset): The ASL dataset to load data from.
    train_size (float, optional): The proportion of the dataset to
    include in the training set. Defaults to 0.8. valid_size (float,
    optional): The proportion of the dataset to include in the
    validation set. Defaults to 0.1. test_size (float, optional): The
    proportion of the dataset to include in the testing set. Defaults to
    0.1. batch_size (int, optional): The number of samples per batch to
    load. Defaults to BATCH_SIZE. random_state (int, optional): The seed
    used by the random number generator for shuffling the data. Defaults
    to SEED.

    Returns: tuple of DataLoader: A tuple containing the data loaders
    for training, validation, and testing sets.

    Parameters

    :   -   **asl_dataset** (*ASLDataset*) -- The ASL dataset to load
            data from.

        -   **train_size** (*float*) -- The proportion of the dataset to
            include in the training set.

        -   **valid_size** (*float*) -- The proportion of the dataset to
            include in the validation set.

        -   **test_size** (*float*) -- The proportion of the dataset to
            include in the testing set.

        -   **batch_size** (*int*) -- The number of samples per batch to
            load.

        -   **random_state** (*int*) -- The seed used by the random
            number generator for shuffling the data.

    Returns

    :   A tuple containing the data loaders for training, validation,
        and testing sets.

    Return type

    :   tuple of DataLoader

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[interpolate_missing_values]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[arr]{.pre}]{.n}*, *[[max_gap]{.pre}]{.n}[[=]{.pre}]{.o}[[3]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#data.data_utils.interpolate_missing_values "Permalink to this definition"){.headerlink}

:   This function provides a solution for handling missing values in the
    data array. It interpolates these missing values, filling them with
    plausible values that maintain the overall data integrity. The
    function uses a linear interpolation method that assumes a straight
    line between the two points on either side of the gap. The maximum
    gap size for which interpolation should be performed is also
    configurable.

    AThe function takes two arguments - an array with missing values,
    and a maximum gap size for interpolation. If the size of the gap
    (i.e., number of consecutive missing values) is less than or equal
    to this specified maximum gap size, the function will fill it with
    interpolated values. This ensures that the data maintains its
    continuity without making too far-fetched estimations for larger
    gaps.

    Args:

    :   arr (np.ndarray): Input array with missing values. max_gap (int,
        optional): Maximum gap to fill. Defaults to
        INTEREMOLATE_MISSING.

    Returns:

    :   np.ndarray: Array with missing values interpolated.

    Functionality:

    :   Interpolates missing values in the array. The function fills
        gaps of up to a maximum size with interpolated values,
        maintaining data integrity and continuity.

    Returns

    :   Array with missing values interpolated.

    Return type

    :   np.ndarray

    Parameters

    :   -   **arr** (*np.ndarray*) -- Input array with missing values.

        -   **max_gap** (*int*) -- Maximum gap to fill.

    This function uses linear interpolation to fill the missing values.
    Other forms of interpolation such as polynomial or spline may
    provide better results for specific types of data. It is also worth
    noting that no imputation method can fully recover original data,
    and as such, results should be interpreted with caution when working
    with imputed data.

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[load_relevant_data_subset]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[pq_path]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.load_relevant_data_subset "Permalink to this definition"){.headerlink}

:   This function serves a key role in handling data in our pipeline by
    loading only a subset of the relevant data from a given path. The
    primary purpose of this is to reduce memory overhead when working
    with large datasets. The implementation relies on efficient data
    loading strategies, leveraging the speed of Parquet file format and
    the ability to read in only necessary chunks of data instead of the
    whole dataset.

    The function takes as input a string which represents the path to
    the data file. It makes use of pandas' parquet read function to read
    the data file. This function is particularly suited for reading
    large datasets as it allows for efficient on-disk storage and fast
    query capabilities. The function uses PyArrow library as the engine
    for reading the parquet files which ensures efficient and fast
    reading of data. After reading the data, the function selects the
    relevant subset based on certain criteria, which is task specific.

    Args:

    :   pq_path (str): Path to the data file.

    Returns:

    :   np.ndarray: Subset of the relevant data as a NumPy array.

    Functionality:

    :   Loads a subset of the relevant data from a given path.

    Returns

    :   Subset of the relevant data.

    Return type

    :   np.ndarray

    Parameters

    :   **pq_path** (*str*) -- Path to the data file.

    The function assumes that the data file is in parquet format and the
    necessary libraries for reading parquet files are installed. It also
    assumes that the path provided is a valid path to the data file.

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[preprocess_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[landmarks]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.preprocess_data "Permalink to this definition"){.headerlink}

:   This function preprocesses the input data by applying similar steps
    as the preprocess_data_to_same_size function, but with the
    difference that it does not interpolate missing values. The function
    again targets to adjust the size of the input data to align with the
    INPUT_SIZE. It selects only non-empty frames and follows similar
    strategies of padding, repeating, and pooling the data for size
    alignment.

    Args:

    :   landmarks (np.ndarray): The input array with landmarks data.

    Returns:

    :   Tuple\[np.ndarray, int\]: A tuple containing processed landmark
        data and the final size of the data.

    Parameters

    :   **landmarks** (*np.ndarray*) -- The input array with landmarks
        data.

    Returns

    :   A tuple containing processed landmark data and the final size of
        the data.

    Return type

    :   Tuple\[np.ndarray, int\]

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[preprocess_data_item]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[raw_landmark_path]{.pre}]{.n}*, *[[targets_sign]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.preprocess_data_item "Permalink to this definition"){.headerlink}

:   The function preprocesses landmark data for a single file. The
    process involves applying transformations to raw landmark data to
    convert it into a form more suitable for machine learning models.
    The transformations may include normalization, scaling, etc. The
    target sign associated with the landmark data is also taken as
    input.

    This function is a handy function to process all landmark aequences
    on a particular location. This will come in handy while testing
    where individual sequences may be provided

    Args:

    :   raw_landmark_path: Path to the raw landmark file targets_sign:
        The target sign for the given landmark data

    Returns: dict: A dictionary containing the preprocessed landmarks,
    target, and size.

    Functionality: - The function reads the parquet file and processes
    the data. - It filters columns to include only frame, type,
    landmark_index, x, and y. - The function then filters face mesh
    landmarks and pose landmarks based on the predefined useful
    landmarks. - Landmarks data is pivoted to have a multi-level column
    structure on landmark type and frame sequence ids. - Missing values
    are interpolated using linear interpolation, and any remaining
    missing values are filled with 0. - The function rearranges columns
    and calculates the number of frames in the data. - X and Y
    coordinates are brought together, and a dictionary with the
    processed data is created and returned.

    Parameters

    :   -   **raw_landmark_path** (*str*) -- Path to the raw landmark
            file.

        -   **targets_sign** (*int*) -- The target sign for the given
            landmark data.

    Returns

    :   A dictionary containing the preprocessed landmarks, target, and
        size.

    Return type

    :   dict

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[preprocess_data_to_same_size]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[landmarks]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.preprocess_data_to_same_size "Permalink to this definition"){.headerlink}

:   This function preprocesses the input data to ensure all data arrays
    have the same size, specified by the global INPUT_SIZE variable.
    This uniform size is necessary for subsequent processing and
    analysis stages, particularly those involving machine learning
    models which often require consistent input sizes. The preprocessing
    involves several steps, including handling missing values,
    upsampling, and reshaping arrays. It begins by interpolating any
    missing values, and then it subsets the data by selecting only
    non-empty frames. Various strategies are applied to align the data
    size to the desired INPUT_SIZE, including padding, repeating, and
    pooling the data.

    Args:

    :   landmarks (np.ndarray): The input array with landmarks data.

    Returns:

    :   Tuple\[np.ndarray, int, int, int\]: A tuple containing processed
        landmark data, the set input size, the number of original
        frames, and the number of frames after preprocessing.

    Parameters

    :   **landmarks** (*np.ndarray*) -- The input array with landmarks
        data.

    Returns

    :   A tuple containing processed landmark data, the set input size,
        the number of original frames, and the

    number of frames after preprocessing. :rtype: Tuple\[np.ndarray,
    int, int, int\]

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[preprocess_raw_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[sample]{.pre}]{.n}[[=]{.pre}]{.o}[[100000]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#data.data_utils.preprocess_raw_data "Permalink to this definition"){.headerlink}

:   Preprocesses the raw data, saves it as numpy arrays into processed
    data directory and updates the metadata CSV file.

    This method preprocess_data preprocesses the data for easier and
    faster loading during training time. The data is processed and
    stored in PROCESSED_DATA_DIR if not already done.

    This function is responsible for preprocessing raw data. The primary
    functionality involves converting raw data into a format more
    suitable for the machine learning pipeline, namely NumPy arrays. The
    function operates on a sample of data, allowing for efficient
    processing of large datasets in manageable chunks. Additionally,
    this function also takes care of persisting the preprocessed data
    for future use and updates the metadata accordingly.

    Args: sample (int): Number of samples to preprocess. Default
    is 100000.

    Functionality: - The function reads the metadata CSV file for
    training data to obtain a dictionary that maps target values to
    integer indices. - It then reads the training data CSV file and
    generates the absolute path to locate landmark files. - Next, it
    keeps text signs and their respective indices and initializes a list
    to store the processed data. - The data is then processed and stored
    in the list by iterating over each file path in the training data
    and reading in the parquet file for that file path. - The landmark
    data is then processed and padded to have a length of
    max_seq_length. - Finally, a dictionary with the processed data is
    created and added to the list. - The processed data is saved to disk
    using the np.save method and the saved file is printed.

    Parameters

    :   **sample** (*int,* *optional,* *default: 100000*) -- Number of
        samples to preprocess.

    Returns

    :   None

    ::: {.admonition .note}
    Note

    If the preprocessed data already exists, the function prints
    "Preprocessed data found. Skipping..." and exits.
    :::

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[remove_outlier_or_missing_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[landmark_len_dict]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.data_utils.remove_outlier_or_missing_data "Permalink to this definition"){.headerlink}

:   This function removes rows from the training data that contain
    missing or outlier landmark data. It takes as input a dictionary
    containing the statistics of landmark lengths for each sign type.
    The function processes the training data and removes rows with
    missing or outlier landmark data. The function also includes a
    nested function 'has_consecutive_zeros' which checks for consecutive
    frames where X and Y coordinates are both zero. If a cleansing
    marker file exists, it skips the process, indicating that the data
    is already cleaned.

    Functionality:

    :   This function takes a dictionary with the statistics of landmark
        lengths per sign type and uses it to identify outlier sequences.
        It removes any rows with missing or outlier landmark data. An
        outlier sequence is defined as one that is either less than a
        third of the median length or more than two standard deviations
        away from the mean length. A row is also marked for deletion if
        the corresponding landmark file is missing or if the sign's
        left-hand or right-hand landmarks contain more than a specified
        number of consecutive zeros.

    Args:

    :   landmark_len_dict (dict): A dictionary containing the statistics
        of landmark lengths for each sign type.

    Returns:

    :   None

    Parameters

    :   **landmark_len_dict** (*dict*) -- A dictionary containing the
        statistics of landmark lengths for each sign type.

    Returns

    :   None, the function doesn't return anything. It modifies data
        in-place.

```{=html}
<!-- -->
```

[[data.data_utils.]{.pre}]{.sig-prename .descclassname}[[remove_unusable_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data.data_utils.remove_unusable_data "Permalink to this definition"){.headerlink}

:   This function checks the existing training data for unusable
    instances, like missing files or data that is smaller than the set
    minimum sequence length. If unusable data is found, it is removed
    from the system, both in terms of files and entries in the training
    dataframe. The dataframe is updated and saved back to the disk. If a
    cleansing marker file exists, it skips the process, indicating that
    the data is already cleaned.

    Functionality:

    :   The function iterates through the DataFrame rows, attempting to
        load and check each landmark file specified in the row's path.
        If the file is missing or if the file's usable size is less than
        a predefined threshold, the function deletes the corresponding
        landmark file and marks the row for deletion in the DataFrame.
        At the end, the function removes all marked rows from the
        DataFrame, updates it and saves it to the disk.

    Returns:

    :   None

    Returns

    :   None, the function doesn't return anything. It modifies data
        in-place.
:::
:::

[]{#document-hparam_search}

::: {#module-hparam_search .section}
[]{#hyperparameter-search}

## HyperParameter Search[¶](#module-hparam_search "Permalink to this headline"){.headerlink}

Examples using MLfowLoggerCallback and setup_mlflow.

*[class]{.pre}[ ]{.w}*[[hparam_search.]{.pre}]{.sig-prename .descclassname}[[Trainer_HparamSearch]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[modelname=\'YetAnotherTransformerClassifier\']{.pre}]{.n}*, *[[dataset=\<class]{.pre} [\'data.dataset.ASL_DATASET\'\>]{.pre}]{.n}*, *[[patience=10]{.pre}]{.n}*[)]{.sig-paren}[¶](#hparam_search.Trainer_HparamSearch "Permalink to this definition"){.headerlink}

:   

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[modelname=\'YetAnotherTransformerClassifier\']{.pre}]{.n}*, *[[dataset=\<class]{.pre} [\'data.dataset.ASL_DATASET\'\>]{.pre}]{.n}*, *[[patience=10]{.pre}]{.n}*[)]{.sig-paren}[¶](#hparam_search.Trainer_HparamSearch.__init__ "Permalink to this definition"){.headerlink}

    :   Initializes the Trainer class with the specified parameters.

        This method initializes various components needed for the
        training process. This includes the model specified by the model
        name, the dataset with optional data augmentation and dropout,
        data loaders for the training, validation, and test sets, a
        SummaryWriter for logging, and a path for saving model
        checkpoints.

        1.  The method first retrieves the specified model and its
            parameters.

        2.  It then initializes the dataset and the data loaders.

        3.  It sets up metrics for early stopping and a writer for
            logging.

        4.  Finally, it prepares a directory for saving model
            checkpoints.

        Args:

        :   modelname (str): The name of the model to be used for
            training. dataset (Dataset): The dataset to be used.
            patience (int): The number of epochs with no improvement
            after which training will be stopped.
            enableAugmentationDropout (bool): If True, enable data
            augmentation dropout. augmentation_threshold (float): The
            threshold for data augmentation.

        Functionality:

        :   This method initializes various components, such as the
            model, dataset, data loaders, logging writer, and checkpoint
            path, required for the training process.

        Parameters

        :   -   **modelname** (*str*) -- The name of the model for
                training.

            -   **dataset** (*Dataset*) -- The dataset for training.

            -   **patience** (*int*) -- The number of epochs with no
                improvement after which training will be stopped.

            -   **enableAugmentationDropout** (*bool*) -- If True,
                enable data augmentation dropout.

            -   **augmentation_threshold** (*float*) -- The threshold
                for data augmentation.

        Return type

        :   None

        ::: {.admonition .note}
        Note

        This method only initializes the Trainer class. The actual
        training is done by calling the train() method.
        :::

        ::: {.admonition .warning}
        Warning

        Make sure the specified model name corresponds to an actual
        model in your project's models directory.
        :::

    [[train]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[n_epochs]{.pre}]{.n}[[=]{.pre}]{.o}[[50]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#hparam_search.Trainer_HparamSearch.train "Permalink to this definition"){.headerlink}

    :   Trains the model for a specified number of epochs.

        This method manages the main training loop of the model. For
        each epoch, it performs several steps. It first puts the model
        into training mode and loops over the training dataset,
        calculating the loss and accuracy for each batch and optimizing
        the model parameters. It logs these metrics and updates a
        progress bar. At the end of each epoch, it evaluates the model
        on the validation set and checks whether early stopping criteria
        have been met. If the early stopping metric has improved, it
        saves the current model and its parameters. If not, it
        increments a counter and potentially stops training if the
        counter exceeds the allowed patience. Finally, it steps the
        learning rate scheduler and calls any registered callbacks.

        1.  The method first puts the model into training mode and
            initializes some lists and counters.

        2.  Then it enters the main loop over the training data,
            updating the model and logging metrics.

        3.  It evaluates the model on the validation set and checks the
            early stopping criteria.

        4.  If the criteria are met, it saves the model and its
            parameters; if not, it increments a patience counter.

        5.  It steps the learning rate scheduler and calls any
            callbacks.

        Args:

        :   n_epochs (int): The number of epochs for which the model
            should be trained.

        Functionality:

        :   This method coordinates the training of the model over a
            series of epochs, handling batch-wise loss computation,
            backpropagation, optimization, validation, early stopping,
            and model checkpoint saving.

        Parameters

        :   **n_epochs** (*int*) -- Number of epochs for training.

        Returns

        :   None

        Return type

        :   None

        ::: {.admonition .note}
        Note

        This method modifies the state of the model and its optimizer,
        as well as various attributes of the Trainer instance itself.
        :::

        ::: {.admonition .warning}
        Warning

        If you set the patience value too low in the constructor, the
        model might stop training prematurely.
        :::
:::

[]{#document-predict_on_camera}

::: {#module-predict_on_camera .section}
[]{#camera-stream-predictions}

## Camera Stream Predictions[¶](#module-predict_on_camera "Permalink to this headline"){.headerlink}

[[predict_on_camera.]{.pre}]{.sig-prename .descclassname}[[convert_mp_to_df]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[results]{.pre}]{.n}*[)]{.sig-paren}[¶](#predict_on_camera.convert_mp_to_df "Permalink to this definition"){.headerlink}

:   

```{=html}
<!-- -->
```

[[predict_on_camera.]{.pre}]{.sig-prename .descclassname}[[show_camera_feed]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[model]{.pre}]{.n}*, *[[LAST_FRAMES]{.pre}]{.n}[[=]{.pre}]{.o}[[32]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#predict_on_camera.show_camera_feed "Permalink to this definition"){.headerlink}

:   
:::

[]{#document-dataset}

::: {#module-data.dataset .section}
[]{#asl-dataset}

## ASL Dataset[¶](#module-data.dataset "Permalink to this headline"){.headerlink}

::: {#asl-dataset-description .section}
### ASL Dataset description[¶](#asl-dataset-description "Permalink to this headline"){.headerlink}

This file contains the ASL_DATASET class which serves as the dataset
module for American Sign Language (ASL) data. The ASL_DATASET is
designed to load, preprocess, augment, and serve the dataset for model
training and validation. This class provides functionalities such as
loading the dataset from disk, applying transformations, data
augmentation techniques, and an interface to access individual data
samples.

::: {.admonition .note}
Note

This dataset class expects data in a specific format. Detailed
explanations and expectations about input data are provided in
respective method docstrings.
:::

*[class]{.pre}[ ]{.w}*[[data.dataset.]{.pre}]{.sig-prename .descclassname}[[ASL_DATASET]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[metadata_df]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[transform]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[max_seq_length]{.pre}]{.n}[[=]{.pre}]{.o}[[32]{.pre}]{.default_value}*, *[[augment]{.pre}]{.n}[[=]{.pre}]{.o}[[False]{.pre}]{.default_value}*, *[[augmentation_threshold]{.pre}]{.n}[[=]{.pre}]{.o}[[0.1]{.pre}]{.default_value}*, *[[enableDropout]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET "Permalink to this definition"){.headerlink}

:   A dataset class for the ASL dataset.

    The ASL_DATASET class represents a dataset of American Sign Language
    (ASL) gestures, where each gesture corresponds to a word or phrase.
    This class provides functionalities to load the dataset, apply
    transformations, augment the data, and yield individual data samples
    for model training and validation.

    [[\_\_getitem\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[idx]{.pre}]{.n}*[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET.__getitem__ "Permalink to this definition"){.headerlink}

    :   Get an item from the dataset by index.

        This method returns a data sample from the dataset based on a
        provided index. It handles reading of the processed data file,
        applies transformations and augmentations (if set), and pads the
        data to match the maximum sequence length. It returns the
        preprocessed landmarks and corresponding target as a tuple.

        Args:

        :   idx (int): The index of the item to retrieve.

        Returns:

        :   tuple: A tuple containing the landmarks and target for the
            item.

        Functionality:

        :   Get a single item from the dataset.

        Parameters

        :   **idx** (*int*) -- The index of the item to retrieve.

        Returns

        :   A tuple containing the landmarks and target for the item.

        Return type

        :   tuple

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[metadata_df]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[transform]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[max_seq_length]{.pre}]{.n}[[=]{.pre}]{.o}[[32]{.pre}]{.default_value}*, *[[augment]{.pre}]{.n}[[=]{.pre}]{.o}[[False]{.pre}]{.default_value}*, *[[augmentation_threshold]{.pre}]{.n}[[=]{.pre}]{.o}[[0.1]{.pre}]{.default_value}*, *[[enableDropout]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET.__init__ "Permalink to this definition"){.headerlink}

    :   Initialize the ASL dataset.

        This method initializes the dataset and loads the metadata
        necessary for the dataset processing. If no metadata is
        provided, it will load the default processed dataset. It also
        sets the transformation functions, data augmentation parameters,
        and maximum sequence length.

        Args:

        :   metadata_df (pd.DataFrame, optional): A dataframe containing
            the metadata for the dataset. Defaults to None. transform
            (callable, optional): A function/transform to apply to the
            data. Defaults to None. max_seq_length (int, optional): The
            maximum sequence length for the data. Defaults to
            INPUT_SIZE. augment (bool, optional): Whether to apply data
            augmentation. Defaults to False. augmentation_threshold
            (float, optional): Probability of augmentation happening.
            Only if augment == True. Defaults to 0.1. enableDropout
            (bool, optional): Whether to enable the frame dropout
            augmentation. Defaults to True.

        Functionality:

        :   Initializes the dataset with necessary configurations and
            loads the data.

        Parameters

        :   -   **metadata_df** (*pd.DataFrame,* *optional*) -- A
                dataframe containing the metadata for the dataset.

            -   **transform** (*callable,* *optional*) -- A
                function/transform to apply to the data.

            -   **max_seq_length** (*int*) -- The maximum sequence
                length for the data.

            -   **augment** (*bool*) -- Whether to apply data
                augmentation.

            -   **augmentation_threshold** (*float*) -- Probability of
                augmentation happening. Only if augment == True.

            -   **enableDropout** (*bool*) -- Whether to enable the
                frame dropout augmentation.

    [[\_\_len\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET.__len__ "Permalink to this definition"){.headerlink}

    :   Get the length of the dataset.

        This method returns the total number of data samples present in
        the dataset. It's an implementation of the special method
        \_\_len\_\_ in Python, providing a way to use the Python
        built-in function len() on the dataset object.

        Functionality:

        :   Get the length of the dataset.

        Returns:

        :   int: The length of the dataset.

        Returns

        :   The length of the dataset.

        Return type

        :   int

    [[\_\_repr\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET.__repr__ "Permalink to this definition"){.headerlink}

    :   Return a string representation of the ASL dataset.

        This method returns a string that provides an overview of the
        dataset, including the number of participants and total data
        samples. It's an implementation of the special method
        \_\_repr\_\_ in Python, providing a human-readable
        representation of the dataset object.

        Returns:

        :   str: A string representation of the dataset.

        Functionality:

        :   Return a string representation of the dataset.

        Returns

        :   A string representation of the dataset.

        Return type

        :   str

    [[\_\_weakref\_\_]{.pre}]{.sig-name .descname}[¶](#data.dataset.ASL_DATASET.__weakref__ "Permalink to this definition"){.headerlink}

    :   list of weak references to the object (if defined)

    [[load_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data.dataset.ASL_DATASET.load_data "Permalink to this definition"){.headerlink}

    :   Load the data for the ASL dataset.

        This method loads the actual ASL data based on the metadata
        provided during initialization. If no metadata was provided, it
        loads the default processed data. It generates absolute paths to
        locate landmark files, and stores individual metadata lists for
        easy access during data retrieval.

        Functionality:

        :   Loads the data for the dataset.

        Return type

        :   None
:::
:::

[]{#document-dl_utils}

::: {#module-dl_utils .section}
[]{#data-utilities}

## Data Utilities[¶](#module-dl_utils "Permalink to this headline"){.headerlink}

::: {#deep-learning-utils .section}
### Deep Learning Utils[¶](#deep-learning-utils "Permalink to this headline"){.headerlink}

This module provides a set of helper functions that abstract away
specific details of different deep learning frameworks (such as
TensorFlow and PyTorch). These functions allow the main code to run in a
framework-agnostic manner, thus improving code portability and
flexibility.

*[class]{.pre}[ ]{.w}*[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[DatasetWithLen]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[tf_dataset]{.pre}]{.n}*, *[[length]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.DatasetWithLen "Permalink to this definition"){.headerlink}

:   The DatasetWithLen class serves as a wrapper around TensorFlow's
    Dataset object. Its primary purpose is to add a length method to the
    TensorFlow Dataset. This is useful in contexts where it's necessary
    to know the number of batches that a DataLoader will create from a
    dataset, which is a common requirement in many machine learning
    training loops. It also provides an iterator over the dataset, which
    facilitates traversing the dataset for operations such as batch
    creation.

    For instance, this might be used in conjunction with a progress bar
    during training to display the total number of batches. Since
    TensorFlow's Dataset objects don't inherently have a \_\_len\_\_
    method, this wrapper class provides that functionality, augmenting
    the dataset with additional features that facilitate the training
    process.

    Args:

    :   tf_dataset: The TensorFlow dataset to be wrapped. length: The
        length of the dataset.

    Functionality:

    :   Provides a length method and an iterator for a TensorFlow
        dataset.

    Return type

    :   DatasetWithLen object

    Parameters

    :   -   **tf_dataset** -- The TensorFlow dataset to be wrapped.

        -   **length** -- The length of the dataset.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[tf_dataset]{.pre}]{.n}*, *[[length]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.DatasetWithLen.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[\_\_iter\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#dl_utils.DatasetWithLen.__iter__ "Permalink to this definition"){.headerlink}

    :   Returns an iterator for the dataset.

        Returns

        :   iterator for the dataset

    [[\_\_len\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#dl_utils.DatasetWithLen.__len__ "Permalink to this definition"){.headerlink}

    :   Returns the length of the dataset.

        Returns

        :   length of the dataset

    [[\_\_weakref\_\_]{.pre}]{.sig-name .descname}[¶](#dl_utils.DatasetWithLen.__weakref__ "Permalink to this definition"){.headerlink}

    :   list of weak references to the object (if defined)

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[get_PT_Dataset]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataloader]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.get_PT_Dataset "Permalink to this definition"){.headerlink}

:   Retrieve the underlying dataset from a PyTorch data loader.

    Parameters

    :   **dataloader** -- DataLoader object.

    Returns

    :   Dataset object.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[get_TF_Dataset]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataloader]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.get_TF_Dataset "Permalink to this definition"){.headerlink}

:   Retrieve the underlying dataset from a TensorFlow data loader.

    Parameters

    :   **dataloader** -- DatasetWithLen object.

    Returns

    :   Dataset object.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[get_dataloader]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*, *[[batch_size]{.pre}]{.n}[[=]{.pre}]{.o}[[128]{.pre}]{.default_value}*, *[[shuffle]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*, *[[dl_framework]{.pre}]{.n}[[=]{.pre}]{.o}[[\'pytorch\']{.pre}]{.default_value}*, *[[num_workers]{.pre}]{.n}[[=]{.pre}]{.o}[[8]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#dl_utils.get_dataloader "Permalink to this definition"){.headerlink}

:   The get_dataloader function is responsible for creating a DataLoader
    object given a dataset and a few other parameters. A DataLoader is
    an essential component in machine learning projects as it controls
    how data is fed into the model during training. However, different
    deep learning frameworks have their own ways of creating and
    handling DataLoader objects.

    To improve the portability and reusability of the code, this
    function abstracts away these specifics, allowing the user to create
    a DataLoader object without having to worry about the details of the
    underlying framework (TensorFlow or PyTorch). This approach can save
    development time and reduce the risk of bugs or errors.

    Args:

    :   dataset: The dataset to be loaded. batch_size: The size of the
        batches that the DataLoader should create. shuffle: Whether to
        shuffle the data before creating batches. dl_framework: The name
        of the deep learning framework. num_workers: The number of
        worker threads to use for loading data.

    Functionality:

    :   Creates and returns a DataLoader object that is compatible with
        the specified deep learning framework.

    Return type

    :   DataLoader or DatasetWithLen object

    Parameters

    :   -   **dataset** -- The dataset to be loaded.

        -   **batch_size** -- The size of the batches that the
            DataLoader should create.

        -   **shuffle** -- Whether to shuffle the data before creating
            batches.

        -   **dl_framework** -- The name of the deep learning framework.

        -   **num_workers** -- The number of worker threads to use for
            loading data.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[get_dataset]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataloader]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.get_dataset "Permalink to this definition"){.headerlink}

:   The get_dataset function is an interface to extract the underlying
    dataset from a dataloader, irrespective of the deep learning
    framework being used, i.e., TensorFlow or PyTorch. The versatility
    of this function makes it integral to any pipeline designed to be
    flexible across both TensorFlow and PyTorch frameworks.

    Given a dataloader object, this function first determines the deep
    learning framework currently in use by referring to the DL_FRAMEWORK
    config parameter variable. If the framework is TensorFlow, it
    invokes the get_TF_Dataset function to retrieve the dataset.
    Alternatively, if PyTorch is being used, the get_PT_Dataset function
    is called. This abstracts away the intricacies of handling different
    deep learning frameworks, thereby simplifying the process of working
    with datasets across TensorFlow and PyTorch.

    Args:

    :   dataloader: DataLoader from PyTorch or DatasetWithLen from
        TensorFlow.

    Functionality:

    :   Extracts the underlying dataset from a dataloader, be it from
        PyTorch or TensorFlow.

    Return type

    :   Dataset object

    Parameters

    :   **dataloader** -- DataLoader in case of PyTorch and
        DatasetWithLen in case of TensorFlow.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[get_model_params]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[model_name]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.get_model_params "Permalink to this definition"){.headerlink}

:   The get_model_params function is a utility function that serves to
    abstract away the details of reading model configurations from a
    YAML file. In a machine learning project, it is common to have
    numerous models, each with its own set of hyperparameters. These
    hyperparameters can be stored in a YAML file for easy access and
    modification.

    This function reads the configuration file and retrieves the
    specific parameters associated with the given model. The
    configurations are stored in a dictionary which is then returned.
    This aids in maintaining a cleaner, more organized codebase and
    simplifies the process of updating or modifying model parameters.

    Args:

    :   model_name: Name of the model whose parameters are to be
        retrieved.

    Functionality:

    :   Reads a YAML file and retrieves the model parameters as a
        dictionary.

    Return type

    :   dict

    Parameters

    :   **model_name** -- Name of the model whose parameters are to be
        retrieved.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[log_metrics]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[phase]{.pre}]{.n}*, *[[loss]{.pre}]{.n}*, *[[acc]{.pre}]{.n}*, *[[epoch]{.pre}]{.n}*, *[[lr]{.pre}]{.n}*, *[[writer]{.pre}]{.n}*[)]{.sig-paren}[¶](#dl_utils.log_metrics "Permalink to this definition"){.headerlink}

:   Helper function to log metrics to TensorBoard.

    Parameters

    :   -   **phase** -- String, the phase of the process ('train' or
            'validation').

        -   **loss** -- Float, the current loss value.

        -   **acc** -- Float, the current accuracy value.

        -   **epoch** -- Integer, the current epoch number.

        -   **lr** -- Float, the current learning rate.

        -   **writer** -- TensorBoard writer object.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[to_PT_DataLoader]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*, *[[batch_size]{.pre}]{.n}[[=]{.pre}]{.o}[[128]{.pre}]{.default_value}*, *[[shuffle]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*, *[[num_workers]{.pre}]{.n}[[=]{.pre}]{.o}[[8]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#dl_utils.to_PT_DataLoader "Permalink to this definition"){.headerlink}

:   This function is the PyTorch counterpart to 'to_TF_DataLoader'. It
    converts a given dataset into a PyTorch DataLoader. The purpose of
    this function is to streamline the creation of PyTorch DataLoaders,
    allowing for easy utilization in a PyTorch training or inference
    pipeline.

    The PyTorch DataLoader handles the process of drawing batches of
    data from a dataset, which is essential when training models. This
    function further extends this functionality by implementing data
    shuffling and utilizing multiple worker threads for asynchronous
    data loading, thereby optimizing the data loading process during
    model training.

    Args:

    :   dataset: The dataset to be loaded. batch_size: The size of each
        batch the DataLoader will return. shuffle: Whether the data
        should be shuffled before batching. num_workers: The number of
        worker threads to use for data loading.

    Functionality:

    :   Converts a given dataset into a PyTorch DataLoader.

    Return type

    :   DataLoader object

    Parameters

    :   -   **dataset** -- The dataset to be loaded.

        -   **batch_size** -- The size of each batch the DataLoader will
            return.

        -   **shuffle** -- Whether the data should be shuffled before
            batching.

        -   **num_workers** -- The number of worker threads to use for
            data loading.

```{=html}
<!-- -->
```

[[dl_utils.]{.pre}]{.sig-prename .descclassname}[[to_TF_DataLoader]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*, *[[batch_size]{.pre}]{.n}[[=]{.pre}]{.o}[[128]{.pre}]{.default_value}*, *[[shuffle]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#dl_utils.to_TF_DataLoader "Permalink to this definition"){.headerlink}

:   This function takes in a dataset and converts it into a TensorFlow
    DataLoader. Its purpose is to provide a streamlined method to
    generate DataLoaders that can be utilized in a TensorFlow training
    or inference pipeline. It not only ensures the dataset is in a
    format that can be ingested by TensorFlow's pipeline, but also
    implements optional shuffling of data, which is a common practice in
    model training to ensure random distribution of data across batches.

    This function first checks whether the data is already in a tensor
    format, if not it converts the data to a tensor. Next, it either
    shuffles the dataset or keeps it as is, based on the 'shuffle' flag.
    Lastly, it prepares the TensorFlow DataLoader by batching the
    dataset and applying an automatic optimization strategy for the
    number of parallel calls in mapping functions.

    Args:

    :   dataset: The dataset to be loaded. batch_size: The size of each
        batch the DataLoader will return. shuffle: Whether the data
        should be shuffled before batching.

    Functionality:

    :   Converts a given dataset into a TensorFlow DataLoader.

    Return type

    :   DatasetWithLen object

    Parameters

    :   -   **dataset** -- The dataset to be loaded.

        -   **batch_size** -- The size of each batch the DataLoader will
            return.

        -   **shuffle** -- Whether the data should be shuffled before
            batching.
:::
:::

[]{#document-trainer}

::: {#module-trainer .section}
[]{#model-training}

## Model Training[¶](#module-trainer "Permalink to this headline"){.headerlink}

::: {#generic-trainer-description .section}
### Generic trainer description[¶](#generic-trainer-description "Permalink to this headline"){.headerlink}

Trainer module handles the training, validation, and testing of
framework-agnostic deep learning models.

The Trainer class handles the complete lifecycle of model training
including setup, execution of training epochs, validation and testing,
early stopping, and result logging.

The class uses configurable parameters for defining training settings
like early stopping and batch size, and it supports adding custom
callback functions to be executed at the end of each epoch. This makes
the trainer class flexible and adaptable for various types of deep
learning models and tasks.

Attributes: model_name (str): The name of the model to be trained.
params (dict): The parameters required for the model. model (model
object): The model object built using the given model name and
parameters. train_loader, valid_loader, test_loader (DataLoader
objects): PyTorch dataloaders for training, validation, and testing
datasets. patience (int): The number of epochs to wait before stopping
training when the validation loss is no longer improving.
best_val_metric (float): The best validation metric recorded.
patience_counter (int): A counter that keeps track of the number of
epochs since the validation loss last improved. model_class (str): The
class name of the model. train_start_time (str): The starting time of
the training process. writer (SummaryWriter object): TensorBoard's
SummaryWriter to log metrics for visualization. checkpoint_path (str):
The path where the best model checkpoints will be saved during training.
epoch (int): The current epoch number. callbacks (list): A list of
callback functions to be called at the end of each epoch.

Methods: train(n_epochs): Trains the model for a specified number of
epochs. evaluate(): Evaluates the model on the validation set. test():
Tests the model on the test set. add_callback(callback): Adds a callback
function to the list of functions to be called at the end of each epoch.

*[class]{.pre}[ ]{.w}*[[trainer.]{.pre}]{.sig-prename .descclassname}[[Trainer]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[modelname=\'YetAnotherTransformerClassifier\']{.pre}]{.n}*, *[[dataset=\<class]{.pre} [\'data.dataset.ASL_DATASET\'\>]{.pre}]{.n}*, *[[patience=10]{.pre}]{.n}*, *[[enableAugmentationDropout=True]{.pre}]{.n}*, *[[augmentation_threshold=0.35]{.pre}]{.n}*[)]{.sig-paren}[¶](#trainer.Trainer "Permalink to this definition"){.headerlink}

:   A trainer class which acts as a control hub for the model lifecycle,
    including initial setup, executing training epochs, performing
    validation and testing, implementing early stopping, and logging
    results. The module has been designed to be agnostic to the specific
    deep learning framework, enhancing its versatility across various
    projects.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[modelname=\'YetAnotherTransformerClassifier\']{.pre}]{.n}*, *[[dataset=\<class]{.pre} [\'data.dataset.ASL_DATASET\'\>]{.pre}]{.n}*, *[[patience=10]{.pre}]{.n}*, *[[enableAugmentationDropout=True]{.pre}]{.n}*, *[[augmentation_threshold=0.35]{.pre}]{.n}*[)]{.sig-paren}[¶](#trainer.Trainer.__init__ "Permalink to this definition"){.headerlink}

    :   Initializes the Trainer class with the specified parameters.

        This method initializes various components needed for the
        training process. This includes the model specified by the model
        name, the dataset with optional data augmentation and dropout,
        data loaders for the training, validation, and test sets, a
        SummaryWriter for logging, and a path for saving model
        checkpoints.

        1.  The method first retrieves the specified model and its
            parameters.

        2.  It then initializes the dataset and the data loaders.

        3.  It sets up metrics for early stopping and a writer for
            logging.

        4.  Finally, it prepares a directory for saving model
            checkpoints.

        Args:

        :   modelname (str): The name of the model to be used for
            training. dataset (Dataset): The dataset to be used.
            patience (int): The number of epochs with no improvement
            after which training will be stopped.
            enableAugmentationDropout (bool): If True, enable data
            augmentation dropout. augmentation_threshold (float): The
            threshold for data augmentation.

        Functionality:

        :   This method initializes various components, such as the
            model, dataset, data loaders, logging writer, and checkpoint
            path, required for the training process.

        Parameters

        :   -   **modelname** (*str*) -- The name of the model for
                training.

            -   **dataset** (*Dataset*) -- The dataset for training.

            -   **patience** (*int*) -- The number of epochs with no
                improvement after which training will be stopped.

            -   **enableAugmentationDropout** (*bool*) -- If True,
                enable data augmentation dropout.

            -   **augmentation_threshold** (*float*) -- The threshold
                for data augmentation.

        Return type

        :   None

        ::: {.admonition .note}
        Note

        This method only initializes the Trainer class. The actual
        training is done by calling the train() method.
        :::

        ::: {.admonition .warning}
        Warning

        Make sure the specified model name corresponds to an actual
        model in your project's models directory.
        :::

    [[\_\_weakref\_\_]{.pre}]{.sig-name .descname}[¶](#trainer.Trainer.__weakref__ "Permalink to this definition"){.headerlink}

    :   list of weak references to the object (if defined)

    [[add_callback]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[callback]{.pre}]{.n}*[)]{.sig-paren}[¶](#trainer.Trainer.add_callback "Permalink to this definition"){.headerlink}

    :   Adds a callback to the Trainer.

        This method simply appends a callback function to the list of
        callbacks stored by the Trainer instance. These callbacks are
        called at the end of each training epoch.

        Functionality:

        :   It allows the addition of custom callbacks to the training
            process, enhancing its flexibility.

        Parameters

        :   **callback** (*Callable*) -- The callback function to be
            added.

        Returns

        :   None

        Return type

        :   None

        ::: {.admonition .warning}
        Warning

        The callback function must be callable and should not modify the
        training process.
        :::

    [[evaluate]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#trainer.Trainer.evaluate "Permalink to this definition"){.headerlink}

    :   Evaluates the model on the validation set.

        This method sets the model to evaluation mode and loops over the
        validation dataset, computing the loss and accuracy for each
        batch. It then averages these metrics and logs them. This
        process provides an unbiased estimate of the model's performance
        on new data during training.

        Functionality:

        :   It manages the evaluation of the model on the validation
            set, handling batch-wise loss computation and accuracy
            assessment.

        Returns

        :   Average validation loss and accuracy

        Return type

        :   Tuple\[float, float\]

        ::: {.admonition .warning}
        Warning

        Ensure the model is in evaluation mode to correctly compute the
        validation metrics.
        :::

    [[test]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#trainer.Trainer.test "Permalink to this definition"){.headerlink}

    :   Tests the model on the test set.

        This method loads the best saved model, sets it to evaluation
        mode, and then loops over the test dataset, computing the loss,
        accuracy, and predictions for each batch. It then averages the
        loss and accuracy and logs them. It also collects all the
        model's predictions and their corresponding labels.

        Functionality:

        :   It manages the testing of the model on the test set,
            handling batch-wise loss computation, accuracy assessment,
            and prediction generation.

        Returns

        :   List of all predictions and their corresponding labels

        Return type

        :   Tuple\[List, List\]

    [[train]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[n_epochs]{.pre}]{.n}[[=]{.pre}]{.o}[[50]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#trainer.Trainer.train "Permalink to this definition"){.headerlink}

    :   Trains the model for a specified number of epochs.

        This method manages the main training loop of the model. For
        each epoch, it performs several steps. It first puts the model
        into training mode and loops over the training dataset,
        calculating the loss and accuracy for each batch and optimizing
        the model parameters. It logs these metrics and updates a
        progress bar. At the end of each epoch, it evaluates the model
        on the validation set and checks whether early stopping criteria
        have been met. If the early stopping metric has improved, it
        saves the current model and its parameters. If not, it
        increments a counter and potentially stops training if the
        counter exceeds the allowed patience. Finally, it steps the
        learning rate scheduler and calls any registered callbacks.

        1.  The method first puts the model into training mode and
            initializes some lists and counters.

        2.  Then it enters the main loop over the training data,
            updating the model and logging metrics.

        3.  It evaluates the model on the validation set and checks the
            early stopping criteria.

        4.  If the criteria are met, it saves the model and its
            parameters; if not, it increments a patience counter.

        5.  It steps the learning rate scheduler and calls any
            callbacks.

        Args:

        :   n_epochs (int): The number of epochs for which the model
            should be trained.

        Functionality:

        :   This method coordinates the training of the model over a
            series of epochs, handling batch-wise loss computation,
            backpropagation, optimization, validation, early stopping,
            and model checkpoint saving.

        Parameters

        :   **n_epochs** (*int*) -- Number of epochs for training.

        Returns

        :   None

        Return type

        :   None

        ::: {.admonition .note}
        Note

        This method modifies the state of the model and its optimizer,
        as well as various attributes of the Trainer instance itself.
        :::

        ::: {.admonition .warning}
        Warning

        If you set the patience value too low in the constructor, the
        model might stop training prematurely.
        :::
:::
:::

[]{#document-visualizations}

::: {#module-visualizations .section}
[]{#data-visualizations}

## Data Visualizations[¶](#module-visualizations "Permalink to this headline"){.headerlink}

[[visualizations.]{.pre}]{.sig-prename .descclassname}[[visualize_data_distribution]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*[)]{.sig-paren}[¶](#visualizations.visualize_data_distribution "Permalink to this definition"){.headerlink}

:   Visualize the distribution of data in terms of the number of samples
    and average sequence length per class.

    This function generates two bar charts: one showing the number of
    samples per class, and the other showing the average sequence length
    per class.

    Parameters

    :   **dataset** (*ASL_Dataset*) -- The ASL dataset to load data
        from.

```{=html}
<!-- -->
```

[[visualizations.]{.pre}]{.sig-prename .descclassname}[[visualize_target_sign]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dataset]{.pre}]{.n}*, *[[target_sign]{.pre}]{.n}*, *[[n_samples]{.pre}]{.n}[[=]{.pre}]{.o}[[6]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#visualizations.visualize_target_sign "Permalink to this definition"){.headerlink}

:   Visualize n_samples instances of a given target sign from the
    dataset.

    This function generates a visual representation of the landmarks for
    each sample belonging to the specified target_sign.

    Args:

    :   dataset (ASL_Dataset): The ASL dataset to load data from.
        target_sign (int): The target sign to visualize. n_samples (int,
        optional): The number of samples to visualize. Defaults to 6.

    Returns:

    :   matplotlib.animation.FuncAnimation: A matplotlib animation
        object displaying the landmarks for each frame.

    Parameters

    :   -   **dataset** (*ASL_Dataset*) -- The ASL dataset to load data
            from.

        -   **target_sign** (*int*) -- The target sign to visualize.

        -   **n_samples** (*int,* *optional*) -- The number of samples
            to visualize, defaults to 6.

    Returns

    :   A matplotlib animation object displaying the landmarks for each
        frame.

    Return type

    :   matplotlib.animation.FuncAnimation
:::

[]{#document-pytorch_models}

::: {#module-models.pytorch.models .section}
[]{#pytorch-models}

## Pytorch Models[¶](#module-models.pytorch.models "Permalink to this headline"){.headerlink}

This kodule defines a PyTorch BaseModel providing a basic framework for
learning and validating from Trainer module, from which other pytorch
models are inherited. This module includes several model classes that
build upon the PyTorch's nn.Module for constructing pytorch LSTM or
Transformer based models:

+-----------------+-----------------------------------------------------+
| Class           | Description                                         |
+=================+=====================================================+
| TransformerSeq  | This is a transformer-based sequence classification |
| uenceClassifier | model. The class constructs a transformer encoder   |
|                 | based on user-defined parameters or default         |
|                 | settings. The forward method first checks and       |
|                 | reshapes the input, then passes it through the      |
|                 | transformer layers. It then pools the sequence by   |
|                 | taking the mean over the time dimension, and        |
|                 | finally applies the output layer to generate the    |
|                 | class predictions.                                  |
+-----------------+-----------------------------------------------------+
| Trans           | A TransformerPredictor model that extends the       |
| formerPredictor | Pytorch BaseModel. This class wraps                 |
|                 | TransformerSequenceClassifier model and provides    |
|                 | functionality to use it for making predictions.     |
+-----------------+-----------------------------------------------------+
| MultiHe         | This class applies a multi-head attention           |
| adSelfAttention | mechanism. It has options for causal masking and    |
|                 | layer normalization. The input is expected to have  |
|                 | dimensions \[batch_size, seq_len, features\].       |
+-----------------+-----------------------------------------------------+
| T               | This class represents a single block of a           |
| ransformerBlock | transformer architecture, including multi-head      |
|                 | self-attention and a feed-forward neural network,   |
|                 | both with optional layer normalization and dropout. |
|                 | The input is expected to have dimensions            |
|                 | \[batch_size, seq_len, features\].                  |
+-----------------+-----------------------------------------------------+
| Y               | This class constructs a transformer-based           |
| etAnotherTransf | classifier with a specified number of               |
| ormerClassifier | TransformerBlock instances. The output of the model |
|                 | is a tensor of logits with dimensions \[batch_size, |
|                 | num_classes\].                                      |
+-----------------+-----------------------------------------------------+
| YetAno          | This class is a wrapper for                         |
| therTransformer | YetAnotherTransformerClassifier which includes      |
|                 | learning rate, optimizer, and learning rate         |
|                 | scheduler settings. It extends from the BaseModel   |
|                 | class.                                              |
+-----------------+-----------------------------------------------------+
| Yet             | This class constructs an ensemble of                |
| AnotherEnsemble | YetAnotherTransformerClassifier instances, where    |
|                 | the outputs are concatenated and passed through a   |
|                 | fully connected layer. This class also extends from |
|                 | the BaseModel class and includes learning rate,     |
|                 | optimizer, and learning rate scheduler settings.    |
+-----------------+-----------------------------------------------------+

: [Model
Classes]{.caption-text}[¶](#id1 "Permalink to this table"){.headerlink}

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[BaseModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel "Permalink to this definition"){.headerlink}

:   A BaseModel that extends the nn.Module from PyTorch.

    Functionality: \#. The class initializes with a given learning rate
    and number of classes. \#. It sets up the loss criterion, accuracy
    metric, and default states for optimizer and scheduler. \#. It
    defines an abstract method 'forward' which should be implemented in
    the subclass. \#. It also defines various utility functions like
    calculating accuracy, training, validation and testing steps,
    scheduler stepping, and model checkpointing.

    Args:

    :   learning_rate (float): The initial learning rate for optimizer.
        n_classes (int): The number of classes for classification.

    Parameters

    :   -   **learning_rate** (*float*) -- The initial learning rate for
            optimizer.

        -   **n_classes** (*int*) -- The number of classes for
            classification.

    Returns

    :   None

    Return type

    :   None

    ::: {.admonition .note}
    Note

    The class does not directly initialize the optimizer and scheduler.
    They should be initialized in the subclass if needed.
    :::

    ::: {.admonition .warning}
    Warning

    The 'forward' function must be implemented in the subclass, else it
    will raise a NotImplementedError.
    :::

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[learning_rate]{.pre}]{.n}*, *[[n_classes]{.pre}]{.n}[[=]{.pre}]{.o}[[250]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[calculate_accuracy]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[y_hat]{.pre}]{.n}*, *[[y]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.calculate_accuracy "Permalink to this definition"){.headerlink}

    :   Calculates the accuracy of the model's prediction.

        Parameters

        :   -   **y_hat** (*Tensor*) -- The predicted output from the
                model.

            -   **y** (*Tensor*) -- The ground truth or actual labels.

        Returns

        :   The calculated accuracy.

        Return type

        :   Tensor

    [[eval_mode]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.eval_mode "Permalink to this definition"){.headerlink}

    :   Sets the model to evaluation mode.

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

    [[get_lr]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.get_lr "Permalink to this definition"){.headerlink}

    :   Gets the current learning rate of the model.

        Returns

        :   The current learning rate.

        Return type

        :   float

    [[load_checkpoint]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[filepath]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.load_checkpoint "Permalink to this definition"){.headerlink}

    :   Loads the model and optimizer states from a checkpoint.

        Parameters

        :   **filepath** (*str*) -- The file path where to load the
            model checkpoint from.

    [[optimize]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.optimize "Permalink to this definition"){.headerlink}

    :   Steps the optimizer and sets the gradients of all optimized
        `torch.Tensor`{.xref .py .py-class .docutils .literal
        .notranslate} s to zero.

    [[save_checkpoint]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[filepath]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.save_checkpoint "Permalink to this definition"){.headerlink}

    :   Saves the model and optimizer states to a checkpoint.

        Parameters

        :   **filepath** (*str*) -- The file path where to save the
            model checkpoint.

    [[step_scheduler]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.step_scheduler "Permalink to this definition"){.headerlink}

    :   Steps the learning rate scheduler, adjusting the optimizer's
        learning rate as necessary.

    [[test_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.test_step "Permalink to this definition"){.headerlink}

    :   Performs a test step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss, accuracy, and model predictions.

        Return type

        :   tuple

    [[train_mode]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.train_mode "Permalink to this definition"){.headerlink}

    :   Sets the model to training mode.

    [[training_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.training_step "Permalink to this definition"){.headerlink}

    :   Performs a training step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss and accuracy.

        Return type

        :   tuple

    [[validation_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.BaseModel.validation_step "Permalink to this definition"){.headerlink}

    :   Performs a validation step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss and accuracy.

        Return type

        :   tuple

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[CVTransferLearningModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.CVTransferLearningModel "Permalink to this definition"){.headerlink}

:   A CVTransferLearningModel that extends the Pytorch BaseModel.

    This class applies transfer learning for computer vision tasks using
    pretrained models. It also provides a forward method to pass an
    input through the model.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    model[nn.Module]{.classifier}

    :   The base model for transfer learning.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.CVTransferLearningModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.CVTransferLearningModel.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[HybridEnsembleModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridEnsembleModel "Permalink to this definition"){.headerlink}

:   A HybridEnsembleModel that extends the Pytorch BaseModel.

    This class creates an ensemble of LSTM and Transformer models and
    provides functionality to use the ensemble for making predictions.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    lstms[nn.ModuleList]{.classifier}

    :   The list of LSTM models.

    models[nn.ModuleList]{.classifier}

    :   The list of Transformer models.

    fc[nn.Linear]{.classifier}

    :   The final fully-connected layer.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridEnsembleModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridEnsembleModel.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[HybridModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridModel "Permalink to this definition"){.headerlink}

:   A HybridModel that extends the Pytorch BaseModel.

    This class combines the LSTMClassifier and
    TransformerSequenceClassifier models and provides functionality to
    use the combined model for making predictions.

    lstm[LSTMClassifier]{.classifier}

    :   The LSTM classifier used for making predictions.

    transformer[TransformerSequenceClassifier]{.classifier}

    :   The transformer sequence classifier used for making predictions.

    fc[nn.Linear]{.classifier}

    :   The final fully-connected layer.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.HybridModel.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[LSTMClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMClassifier "Permalink to this definition"){.headerlink}

:   A LSTM-based Sequence Classifier. This class utilizes a LSTM network
    for sequence classification tasks.

    DEFAULTS[dict]{.classifier}

    :   Default settings for the LSTM and classifier. These can be
        overridden by passing values in the constructor.

    lstm[nn.LSTM]{.classifier}

    :   The LSTM network used for processing the input sequence.

    dropout[nn.Dropout]{.classifier}

    :   The dropout layer applied after LSTM network.

    output_layer[nn.Linear]{.classifier}

    :   The output layer used to generate class predictions.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMClassifier.forward "Permalink to this definition"){.headerlink}

    :   Forward pass through the model

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[LSTMPredictor]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMPredictor "Permalink to this definition"){.headerlink}

:   A LSTMPredictor model that extends the Pytorch BaseModel.

    This class wraps the LSTMClassifier model and provides functionality
    to use it for making predictions.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    model[LSTMClassifier]{.classifier}

    :   The LSTM classifier used for making predictions.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMPredictor.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.LSTMPredictor.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[MultiHeadSelfAttention]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.MultiHeadSelfAttention "Permalink to this definition"){.headerlink}

:   A MultiHeadSelfAttention module that extends the nn.Module from
    PyTorch.

    Functionality: \#. The class initializes with a given dimension
    size, number of attention heads, dropout rate, layer normalization
    and causality. \#. It sets up the multihead attention module and
    layer normalization. \#. It also defines a forward method that
    applies the multihead attention, causal masking if requested, and
    layer normalization if requested.

    multihead_attn[nn.MultiheadAttention]{.classifier}

    :   The multihead attention module.

    layer_norm[nn.LayerNorm or None]{.classifier}

    :   The layer normalization module. If it is not applied, set to
        None.

    causal[bool]{.classifier}

    :   If True, applies causal masking.

    forward(x)

    :   Performs a forward pass through the model.

    Args:

    :   dim (int): The dimension size of the input data. num_heads
        (int): The number of attention heads. dropout (float): The
        dropout rate. layer_norm (bool): Whether to apply layer
        normalization. causal (bool): Whether to apply causal masking.

    Returns: None

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dim]{.pre}]{.n}*, *[[num_heads]{.pre}]{.n}[[=]{.pre}]{.o}[[8]{.pre}]{.default_value}*, *[[dropout]{.pre}]{.n}[[=]{.pre}]{.o}[[0.1]{.pre}]{.default_value}*, *[[layer_norm]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*, *[[causal]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.pytorch.models.MultiHeadSelfAttention.__init__ "Permalink to this definition"){.headerlink}

    :   

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[TransformerBlock]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerBlock "Permalink to this definition"){.headerlink}

:   A TransformerBlock module that extends the nn.Module from PyTorch.

    Functionality: \#. The class initializes with a given dimension
    size, number of attention heads, expansion factor, attention dropout
    rate, and dropout rate. \#. It sets up the multihead self-attention
    module, layer normalization and feed-forward network. \#. It also
    defines a forward method that applies the multihead self-attention,
    dropout, layer normalization and feed-forward network.

    norm1, norm2, norm3[nn.LayerNorm]{.classifier}

    :   The layer normalization modules.

    attn[MultiHeadSelfAttention]{.classifier}

    :   The multihead self-attention module.

    feed_forward[nn.Sequential]{.classifier}

    :   The feed-forward network.

    dropout[nn.Dropout]{.classifier}

    :   The dropout module.

    forward(x)

    :   Performs a forward pass through the model.

    Args:

    :   dim (int): The dimension size of the input data. num_heads
        (int): The number of attention heads. expansion_factor (int):
        The expansion factor for the hidden layer size in the
        feed-forward network. attn_dropout (float): The dropout rate for
        the attention module. drop_rate (float): The dropout rate for
        the module.

    Returns: None

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[dim]{.pre}]{.n}[[=]{.pre}]{.o}[[192]{.pre}]{.default_value}*, *[[num_heads]{.pre}]{.n}[[=]{.pre}]{.o}[[4]{.pre}]{.default_value}*, *[[expansion_factor]{.pre}]{.n}[[=]{.pre}]{.o}[[4]{.pre}]{.default_value}*, *[[attn_dropout]{.pre}]{.n}[[=]{.pre}]{.o}[[0.2]{.pre}]{.default_value}*, *[[drop_rate]{.pre}]{.n}[[=]{.pre}]{.o}[[0.2]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerBlock.__init__ "Permalink to this definition"){.headerlink}

    :   

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[TransformerEnsemble]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerEnsemble "Permalink to this definition"){.headerlink}

:   A TransformerEnsemble that extends the Pytorch BaseModel.

    This class creates an ensemble of TransformerSequenceClassifier
    models and provides functionality to use the ensemble for making
    predictions.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    models[nn.ModuleList]{.classifier}

    :   The list of transformer sequence classifiers used for making
        predictions.

    fc[nn.Linear]{.classifier}

    :   The final fully-connected layer.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerEnsemble.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerEnsemble.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[TransformerPredictor]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerPredictor "Permalink to this definition"){.headerlink}

:   A TransformerPredictor model that extends the Pytorch BaseModel.

    This class wraps the TransformerSequenceClassifier model and
    provides functionality to use it for making predictions.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    model[TransformerSequenceClassifier]{.classifier}

    :   The transformer sequence classifier used for making predictions.

    optimizer[torch.optim.Adam]{.classifier}

    :   The optimizer used for updating the model parameters.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler used for adapting the learning rate
        during training.

    forward(x)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerPredictor.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerPredictor.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[TransformerSequenceClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerSequenceClassifier "Permalink to this definition"){.headerlink}

:   A Transformer-based Sequence Classifier. This class utilizes a
    transformer encoder to process the input sequence.

    The transformer encoder consists of a stack of N transformer layers
    that are applied to the input sequence. The output sequence from the
    transformer encoder is then passed through a linear layer to
    generate class predictions.

    DEFAULTS[dict]{.classifier}

    :   Default settings for the transformer encoder and classifier.
        These can be overridden by passing values in the constructor.

    transformer[nn.TransformerEncoder]{.classifier}

    :   The transformer encoder used to process the input sequence.

    output_layer[nn.Linear]{.classifier}

    :   The output layer used to generate class predictions.

    batch_first[bool]{.classifier}

    :   Whether the first dimension of the input tensor represents the
        batch size.

    forward(inputs)

    :   Performs a forward pass through the model.

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerSequenceClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.TransformerSequenceClassifier.forward "Permalink to this definition"){.headerlink}

    :   Forward pass through the model

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[YetAnotherEnsemble]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherEnsemble "Permalink to this definition"){.headerlink}

:   A YetAnotherEnsemble model that extends the Pytorch BaseModel.

    Functionality: \#. The class initializes with a set of parameters
    for the YetAnotherTransformerClassifier. \#. It sets up an ensemble
    of YetAnotherTransformerClassifier models, a fully connected layer,
    the optimizer and the learning rate scheduler. \#. It also defines a
    forward method that applies each YetAnotherTransformerClassifier
    model in the ensemble, concatenates the outputs and applies the
    fully connected layer.

    Args:

    :   kwargs (dict): A dictionary containing the parameters for the
        YetAnotherTransformerClassifier models, fully connected layer,
        optimizer and learning rate scheduler.

    Returns: None

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherEnsemble.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherEnsemble.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[YetAnotherTransformer]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformer "Permalink to this definition"){.headerlink}

:   A YetAnotherTransformer model that extends the Pytorch BaseModel.

    Functionality: \#. The class initializes with a set of parameters
    for the YetAnotherTransformerClassifier. \#. It sets up the
    YetAnotherTransformerClassifier model, the optimizer and the
    learning rate scheduler. \#. It also defines a forward method that
    applies the YetAnotherTransformerClassifier model.

    learning_rate[float]{.classifier}

    :   The learning rate for the optimizer.

    model[YetAnotherTransformerClassifier]{.classifier}

    :   The YetAnotherTransformerClassifier model.

    optimizer[torch.optim.AdamW]{.classifier}

    :   The AdamW optimizer.

    scheduler[torch.optim.lr_scheduler.ExponentialLR]{.classifier}

    :   The learning rate scheduler.

    forward(x)

    :   Performs a forward pass through the model.

    Args:

    :   kwargs (dict): A dictionary containing the parameters for the
        YetAnotherTransformerClassifier, optimizer and learning rate
        scheduler.

    Returns: None

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformer.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformer.forward "Permalink to this definition"){.headerlink}

    :   The forward function for the BaseModel.

        Parameters

        :   **x** (*Tensor*) -- The inputs to the model.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.models.]{.pre}]{.sig-prename .descclassname}[[YetAnotherTransformerClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformerClassifier "Permalink to this definition"){.headerlink}

:   A YetAnotherTransformerClassifier module that extends the nn.Module
    from PyTorch.

    Functionality: \#. The class initializes with a set of parameters
    for the transformer blocks. \#. It sets up the transformer blocks
    and the output layer. \#. It also defines a forward method that
    applies the transformer blocks, takes the mean over the time
    dimension of the transformed sequence, and applies the output layer.

    DEFAULTS[dict]{.classifier}

    :   The default settings for the transformer.

    settings[dict]{.classifier}

    :   The settings for the transformer, with any user-provided values
        overriding the defaults.

    transformer[nn.ModuleList]{.classifier}

    :   The list of transformer blocks.

    output_layer[nn.Linear]{.classifier}

    :   The output layer.

    forward(inputs)

    :   Performs a forward pass through the model.

    Args:

    :   kwargs (dict): A dictionary containing the parameters for the
        transformer blocks.

    Returns: None

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformerClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.models.YetAnotherTransformerClassifier.forward "Permalink to this definition"){.headerlink}

    :   Forward pass through the model
:::

[]{#document-tensorflow_models}

::: {#module-models.tensorflow.models .section}
[]{#tensorflow-models}

## Tensorflow Models[¶](#module-models.tensorflow.models "Permalink to this headline"){.headerlink}

This kodule defines a PyTorch BaseModel providing a basic framework for
learning and validating from Trainer

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[BaseModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel "Permalink to this definition"){.headerlink}

:   A BaseModel that extends the tf.keras.Model.

    Functionality: \#. The class initializes with a given learning
    rate. \#. It sets up the loss criterion, accuracy metric, and
    default states for optimizer and scheduler. \#. It defines an
    abstract method 'call' which should be implemented in the
    subclass. \#. It also defines various utility functions like
    calculating accuracy, training, validation and testing steps,
    scheduler stepping, and model checkpointing.

    Args:

    :   learning_rate (float): The initial learning rate for optimizer.

    Parameters

    :   **learning_rate** (*float*) -- The initial learning rate for
        optimizer.

    Returns

    :   None

    Return type

    :   None

    ::: {.admonition .note}
    Note

    The class does not directly initialize the optimizer and scheduler.
    They should be initialized in the subclass if needed.
    :::

    ::: {.admonition .warning}
    Warning

    The 'call' function must be implemented in the subclass, else it
    will raise a NotImplementedError.
    :::

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[learning_rate]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[calculate_accuracy]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[y_pred]{.pre}]{.n}*, *[[y_true]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.calculate_accuracy "Permalink to this definition"){.headerlink}

    :   Calculates the accuracy of the model's prediction.

        Parameters

        :   -   **y_pred** (*Tensor*) -- The predicted output from the
                model.

            -   **y_true** (*Tensor*) -- The ground truth or actual
                labels.

        Returns

        :   The calculated accuracy.

        Return type

        :   float

    [[call]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*, *[[training]{.pre}]{.n}[[=]{.pre}]{.o}[[False]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.call "Permalink to this definition"){.headerlink}

    :   The call function for the BaseModel.

        Parameters

        :   -   **inputs** (*Tensor*) -- The inputs to the model.

            -   **training** (*bool*) -- A flag indicating whether the
                model is in training mode. Default is False.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

    [[eval_mode]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.eval_mode "Permalink to this definition"){.headerlink}

    :   Sets the model to evaluation mode.

    [[get_lr]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.get_lr "Permalink to this definition"){.headerlink}

    :   Gets the current learning rate of the model.

        Returns

        :   The current learning rate.

        Return type

        :   float

    [[load_checkpoint]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[filepath]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.load_checkpoint "Permalink to this definition"){.headerlink}

    :   Loads the model weights from a checkpoint.

        Parameters

        :   **filepath** (*str*) -- The file path where to load the
            model checkpoint from.

    [[optimize]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.optimize "Permalink to this definition"){.headerlink}

    :   Sets the model to training mode.

    [[save_checkpoint]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[filepath]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.save_checkpoint "Permalink to this definition"){.headerlink}

    :   Saves the model weights to a checkpoint.

        Parameters

        :   **filepath** (*str*) -- The file path where to save the
            model checkpoint.

    [[step_scheduler]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.step_scheduler "Permalink to this definition"){.headerlink}

    :   Adjusts the learning rate according to the learning rate
        scheduler.

    [[test_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.test_step "Permalink to this definition"){.headerlink}

    :   Performs a test step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss, accuracy, and model predictions.

        Return type

        :   tuple

    [[train_mode]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.train_mode "Permalink to this definition"){.headerlink}

    :   Sets the model to training mode.

    [[training_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.training_step "Permalink to this definition"){.headerlink}

    :   Performs a training step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss and accuracy.

        Return type

        :   tuple

    [[validation_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.BaseModel.validation_step "Permalink to this definition"){.headerlink}

    :   Performs a validation step using the input batch data.

        Parameters

        :   **batch** (*tuple*) -- A tuple containing input data and
            labels.

        Returns

        :   The calculated loss and accuracy.

        Return type

        :   tuple

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[HybridModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.HybridModel "Permalink to this definition"){.headerlink}

:   

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.HybridModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[call]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*, *[[training]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.tensorflow.models.HybridModel.call "Permalink to this definition"){.headerlink}

    :   The call function for the BaseModel.

        Parameters

        :   -   **inputs** (*Tensor*) -- The inputs to the model.

            -   **training** (*bool*) -- A flag indicating whether the
                model is in training mode. Default is False.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[LSTMClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMClassifier "Permalink to this definition"){.headerlink}

:   LSTM-based Sequence Classifier

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[call]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMClassifier.call "Permalink to this definition"){.headerlink}

    :   Forward pass through the model

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[LSTMPredictor]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMPredictor "Permalink to this definition"){.headerlink}

:   

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMPredictor.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[call]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.LSTMPredictor.call "Permalink to this definition"){.headerlink}

    :   The call function for the BaseModel.

        Parameters

        :   -   **inputs** (*Tensor*) -- The inputs to the model.

            -   **training** (*bool*) -- A flag indicating whether the
                model is in training mode. Default is False.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[TransformerEncoderLayer]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerEncoderLayer "Permalink to this definition"){.headerlink}

:   A Transformer Encoder layer as a subclass of tf.keras.layers.Layer.

    Functionality: \#. The class first initializes with key parameters
    for MultiHeadAttention and feedforward network. \#. Then it defines
    the key components like multi-head attention, feedforward network,
    layer normalization, and dropout. \#. In the call function, it takes
    input and performs self-attention, followed by layer normalization
    and feedforward operation.

    Args:

    :   d_model (int): The dimensionality of the input. n_head (int):
        The number of heads in the multi-head attention. dim_feedforward
        (int): The dimensionality of the feedforward network model.
        dropout (float): The dropout value.

    Parameters

    :   -   **d_model** (*int*) -- The dimensionality of the input.

        -   **n_head** (*int*) -- The number of heads in the multi-head
            attention.

        -   **dim_feedforward** (*int*) -- The dimensionality of the
            feedforward network model.

        -   **dropout** (*float*) -- The dropout value.

    Returns

    :   None

    Return type

    :   None

    ::: {.admonition .note}
    Note

    The implementation is based on the "Attention is All You Need"
    paper.
    :::

    ::: {.admonition .warning}
    Warning

    Ensure that the input dimension 'd_model' is divisible by the number
    of attention heads 'n_head'.
    :::

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[d_model]{.pre}]{.n}*, *[[n_head]{.pre}]{.n}*, *[[dim_feedforward]{.pre}]{.n}*, *[[dropout]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerEncoderLayer.__init__ "Permalink to this definition"){.headerlink}

    :   

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[TransformerPredictor]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerPredictor "Permalink to this definition"){.headerlink}

:   A Transformer Predictor model that extends the BaseModel.

    Functionality: \#. The class first initializes with the learning
    rate and other parameters. \#. It then creates an instance of
    TransformerSequenceClassifier. \#. It also sets up the learning rate
    scheduler and the optimizer. \#. In the call function, it simply
    runs the TransformerSequenceClassifier.

    Args:

    :   kwargs (dict): A dictionary of arguments.

    Parameters

    :   **kwargs** (*dict*) -- A dictionary of arguments.

    Returns

    :   None

    Return type

    :   None

    ::: {.admonition .note}
    Note

    The learning rate is set up with an exponential decay schedule.
    :::

    ::: {.admonition .warning}
    Warning

    The learning rate and gamma for the decay schedule must be specified
    in the 'kwargs'.
    :::

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerPredictor.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[call]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*, *[[training]{.pre}]{.n}[[=]{.pre}]{.o}[[True]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerPredictor.call "Permalink to this definition"){.headerlink}

    :   The call function for the BaseModel.

        Parameters

        :   -   **inputs** (*Tensor*) -- The inputs to the model.

            -   **training** (*bool*) -- A flag indicating whether the
                model is in training mode. Default is False.

        Returns

        :   None

        ::: {.admonition .warning}
        Warning

        This function must be implemented in the subclass, else it
        raises a NotImplementedError.
        :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.tensorflow.models.]{.pre}]{.sig-prename .descclassname}[[TransformerSequenceClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerSequenceClassifier "Permalink to this definition"){.headerlink}

:   A Transformer Sequence Classifier as a subclass of tf.keras.Model.

    Functionality: \#. The class first initializes with default or
    provided settings. \#. Then it defines the key components like the
    transformer encoder layers and output layer. \#. In the call
    function, it takes input and passes it through each transformer
    layer followed by normalization and dense layer for final output.

    Args:

    :   kwargs (dict): Any additional arguments. If not provided,
        defaults will be used.

    Parameters

    :   **kwargs** (*dict*) -- Any additional arguments.

    Returns

    :   None

    Return type

    :   None

    ::: {.admonition .note}
    Note

    The implementation is based on the "Attention is All You Need"
    paper.
    :::

    ::: {.admonition .warning}
    Warning

    The inputs should have a shape of (batch_size, seq_length, height,
    width), otherwise, a ValueError will be raised.
    :::

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.tensorflow.models.TransformerSequenceClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   
:::

[]{#document-lightning_models}

::: {#module-models.pytorch.lightning_models .section}
[]{#torch-lightning-models}

## Torch Lightning Models[¶](#module-models.pytorch.lightning_models "Permalink to this headline"){.headerlink}

*[class]{.pre}[ ]{.w}*[[models.pytorch.lightning_models.]{.pre}]{.sig-prename .descclassname}[[LightningBaseModel]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel "Permalink to this definition"){.headerlink}

:   

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[learning_rate]{.pre}]{.n}*, *[[n_classes]{.pre}]{.n}[[=]{.pre}]{.o}[[250]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[configure_optimizers]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.configure_optimizers "Permalink to this definition"){.headerlink}

    :   Choose what optimizers and learning-rate schedulers to use in
        your optimization. Normally you'd need one. But in the case of
        GANs or similar you might have multiple. Optimization with
        multiple optimizers only works in the manual optimization mode.

        Return:

        :   Any of these 6 options.

            -   **Single optimizer**.

            -   **List or Tuple** of optimizers.

            -   **Two lists** - The first list has multiple optimizers,
                and the second has multiple LR schedulers (or multiple
                `lr_scheduler_config`{.docutils .literal .notranslate}).

            -   **Dictionary**, with an `"optimizer"`{.docutils .literal
                .notranslate} key, and (optionally) a
                `"lr_scheduler"`{.docutils .literal .notranslate} key
                whose value is a single LR scheduler or
                `lr_scheduler_config`{.docutils .literal .notranslate}.

            -   **None** - Fit will run without any optimizer.

        The `lr_scheduler_config`{.docutils .literal .notranslate} is a
        dictionary which contains the scheduler and its associated
        configuration. The default configuration is shown below.

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        :::
        :::

        When there are schedulers in which the `.step()`{.docutils
        .literal .notranslate} method is conditioned on a value, such as
        the `torch.optim.lr_scheduler.ReduceLROnPlateau`{.xref .py
        .py-class .docutils .literal .notranslate} scheduler, Lightning
        requires that the `lr_scheduler_config`{.docutils .literal
        .notranslate} contains the keyword `"monitor"`{.docutils
        .literal .notranslate} set to the metric name that the scheduler
        should be conditioned on.

        Metrics can be made available to monitor by simply logging it
        using `self.log('metric_to_track', metric_val)`{.docutils
        .literal .notranslate} in your `LightningModule`{.xref .py
        .py-class .docutils .literal .notranslate}.

        Note:

        :   Some things to know:

            -   Lightning calls `.backward()`{.docutils .literal
                .notranslate} and `.step()`{.docutils .literal
                .notranslate} automatically in case of automatic
                optimization.

            -   If a learning rate scheduler is specified in
                `configure_optimizers()`{.docutils .literal
                .notranslate} with key `"interval"`{.docutils .literal
                .notranslate} (default "epoch") in the scheduler
                configuration, Lightning will call the scheduler's
                `.step()`{.docutils .literal .notranslate} method
                automatically in case of automatic optimization.

            -   If you use 16-bit precision (`precision=16`{.docutils
                .literal .notranslate}), Lightning will automatically
                handle the optimizer.

            -   If you use `torch.optim.LBFGS`{.xref .py .py-class
                .docutils .literal .notranslate}, Lightning handles the
                closure function automatically for you.

            -   If you use multiple optimizers, you will have to switch
                to 'manual optimization' mode and step them yourself.

            -   If you need to control how often the optimizer steps,
                override the `optimizer_step()`{.xref .py .py-meth
                .docutils .literal .notranslate} hook.

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.forward "Permalink to this definition"){.headerlink}

    :   Same as `torch.nn.Module.forward()`{.xref .py .py-meth .docutils
        .literal .notranslate}.

        Args:

        :   [[\*]{#id2 .problematic}](#id1)args: Whatever you decide to
            pass into the forward method. [[\*\*]{#id4
            .problematic}](#id3)kwargs: Keyword arguments are also
            possible.

        Return:

        :   Your model's output

    [[on_test_end]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[None]{.pre}]{.sig-return-typehint}]{.sig-return}[¶](#models.pytorch.lightning_models.LightningBaseModel.on_test_end "Permalink to this definition"){.headerlink}

    :   Called at the end of testing.

    [[on_train_epoch_end]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[None]{.pre}]{.sig-return-typehint}]{.sig-return}[¶](#models.pytorch.lightning_models.LightningBaseModel.on_train_epoch_end "Permalink to this definition"){.headerlink}

    :   Called in the training loop at the very end of the epoch.

        To access all batch outputs at the end of the epoch, you can
        cache step outputs as an attribute of the
        `LightningModule`{.xref .py .py-class .docutils .literal
        .notranslate} and access them in this hook:

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            class MyLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.training_step_outputs = []

                def training_step(self):
                    loss = ...
                    self.training_step_outputs.append(loss)
                    return loss

                def on_train_epoch_end(self):
                    # do something with all training_step outputs, for example:
                    epoch_mean = torch.stack(self.training_step_outputs).mean()
                    self.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    self.training_step_outputs.clear()
        :::
        :::

    [[on_validation_end]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.on_validation_end "Permalink to this definition"){.headerlink}

    :   Called at the end of validation.

    [[test_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*, *[[batch_idx]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.test_step "Permalink to this definition"){.headerlink}

    :   Operates on a single batch of data from the test set. In this
        step you'd normally generate examples or calculate anything of
        interest such as accuracy.

        Args:

        :   batch: The output of your `DataLoader`{.xref .py .py-class
            .docutils .literal .notranslate}. batch_idx: The index of
            this batch. dataloader_id: The index of the dataloader that
            produced this batch.

            > <div>
            >
            > (only if multiple test dataloaders used).
            >
            > </div>

        Return:

        :   Any of.

            > <div>
            >
            > -   Any object or value
            >
            > -   `None`{.docutils .literal .notranslate} - Testing will
            >     skip to the next batch
            >
            > </div>

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            # if you have one test dataloader:
            def test_step(self, batch, batch_idx):
                ...


            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                ...
        :::
        :::

        Examples:

        ::: {.highlight-default .notranslate}
        ::: {.highlight}
            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        :::
        :::

        If you pass in multiple test dataloaders, [`test_step()`{.xref
        .py .py-meth .docutils .literal
        .notranslate}](#models.pytorch.lightning_models.LightningBaseModel.test_step "models.pytorch.lightning_models.LightningBaseModel.test_step"){.reference
        .internal} will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch
        between single and multiple dataloaders.

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        :::
        :::

        Note:

        :   If you don't need to test you don't need to implement this
            method.

        Note:

        :   When the [`test_step()`{.xref .py .py-meth .docutils
            .literal
            .notranslate}](#models.pytorch.lightning_models.LightningBaseModel.test_step "models.pytorch.lightning_models.LightningBaseModel.test_step"){.reference
            .internal} is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of the
            test epoch, the model goes back to training mode and
            gradients are enabled.

    [[training_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*, *[[batch_idx]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.training_step "Permalink to this definition"){.headerlink}

    :   Here you compute and return the training loss and some
        additional metrics for e.g. the progress bar or logger.

        Args:

        :   

            batch (`Tensor`{.xref .py .py-class .docutils .literal .notranslate} \| (`Tensor`{.xref .py .py-class .docutils .literal .notranslate}, ...) \| \[`Tensor`{.xref .py .py-class .docutils .literal .notranslate}, ...\]):

            :   The output of your `DataLoader`{.xref .py .py-class
                .docutils .literal .notranslate}. A tensor, tuple or
                list.

            batch_idx (`int`{.docutils .literal .notranslate}): Integer
            displaying index of this batch

        Return:

        :   Any of.

            -   `Tensor`{.xref .py .py-class .docutils .literal
                .notranslate} - The loss tensor

            -   `dict`{.docutils .literal .notranslate} - A dictionary.
                Can include any keys, but must include the key
                `'loss'`{.docutils .literal .notranslate}

            -   

                `None`{.docutils .literal .notranslate} - Training will skip to the next batch. This is only for automatic optimization.

                :   This is not supported for multi-GPU, TPU, IPU, or
                    DeepSpeed.

        In this step you'd normally do the forward pass and calculate
        the loss for a batch. You can also do fancier things like
        multiple forward passes or something model specific.

        Example:

        ::: {.highlight-default .notranslate}
        ::: {.highlight}
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss
        :::
        :::

        To use multiple optimizers, you can switch to 'manual
        optimization' and control their stepping:

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()
        :::
        :::

        Note:

        :   When `accumulate_grad_batches`{.docutils .literal
            .notranslate} \> 1, the loss returned here will be
            automatically normalized by
            `accumulate_grad_batches`{.docutils .literal .notranslate}
            internally.

    [[validation_step]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[batch]{.pre}]{.n}*, *[[batch_idx]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningBaseModel.validation_step "Permalink to this definition"){.headerlink}

    :   Operates on a single batch of data from the validation set. In
        this step you'd might generate examples or calculate anything of
        interest like accuracy.

        Args:

        :   batch: The output of your `DataLoader`{.xref .py .py-class
            .docutils .literal .notranslate}. batch_idx: The index of
            this batch. dataloader_idx: The index of the dataloader that
            produced this batch.

            > <div>
            >
            > (only if multiple val dataloaders used)
            >
            > </div>

        Return:

        :   -   Any object or value

            -   `None`{.docutils .literal .notranslate} - Validation
                will skip to the next batch

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx):
                ...


            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                ...
        :::
        :::

        Examples:

        ::: {.highlight-default .notranslate}
        ::: {.highlight}
            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        :::
        :::

        If you pass in multiple val dataloaders,
        [`validation_step()`{.xref .py .py-meth .docutils .literal
        .notranslate}](#models.pytorch.lightning_models.LightningBaseModel.validation_step "models.pytorch.lightning_models.LightningBaseModel.validation_step"){.reference
        .internal} will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch
        between single and multiple dataloaders.

        ::: {.highlight-python .notranslate}
        ::: {.highlight}
            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        :::
        :::

        Note:

        :   If you don't need to validate you don't need to implement
            this method.

        Note:

        :   When the [`validation_step()`{.xref .py .py-meth .docutils
            .literal
            .notranslate}](#models.pytorch.lightning_models.LightningBaseModel.validation_step "models.pytorch.lightning_models.LightningBaseModel.validation_step"){.reference
            .internal} is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of
            validation, the model goes back to training mode and
            gradients are enabled.

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.lightning_models.]{.pre}]{.sig-prename .descclassname}[[LightningTransformerPredictor]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerPredictor "Permalink to this definition"){.headerlink}

:   

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerPredictor.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[x]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerPredictor.forward "Permalink to this definition"){.headerlink}

    :   Same as `torch.nn.Module.forward()`{.xref .py .py-meth .docutils
        .literal .notranslate}.

        Args:

        :   [[\*]{#id6 .problematic}](#id5)args: Whatever you decide to
            pass into the forward method. [[\*\*]{#id8
            .problematic}](#id7)kwargs: Keyword arguments are also
            possible.

        Return:

        :   Your model's output

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[models.pytorch.lightning_models.]{.pre}]{.sig-prename .descclassname}[[LightningTransformerSequenceClassifier]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*]{.pre}]{.o}[[args]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*, *[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}[[:]{.pre}]{.p}[ ]{.w}[[Any]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier "Permalink to this definition"){.headerlink}

:   Transformer-based Sequence Classifier

    [[\_\_init\_\_]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[\*\*]{.pre}]{.o}[[kwargs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier.__init__ "Permalink to this definition"){.headerlink}

    :   

    [[forward]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[inputs]{.pre}]{.n}*[)]{.sig-paren}[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier.forward "Permalink to this definition"){.headerlink}

    :   Forward pass through the model
:::
:::
:::

::: {#indices-and-tables .section}
# Indices and tables[¶](#indices-and-tables "Permalink to this headline"){.headerlink}

-   [[Index]{.std .std-ref}](genindex.html){.reference .internal}

-   [[Module Index]{.std .std-ref}](py-modindex.html){.reference
    .internal}

-   [[Search Page]{.std .std-ref}](search.html){.reference .internal}
:::
:::
:::
:::

::: {.sphinxsidebar role="navigation" aria-label="main navigation"}
::: {.sphinxsidebarwrapper}
# [American Sign Language Recognition](#) {#american-sign-language-recognition .logo}

### Navigation

[Contents:]{.caption-text}

-   [Data Augmentations](index.html#document-augmentations){.reference
    .internal}
-   [Training Callbacks](index.html#document-callbacks){.reference
    .internal}
-   [Project Configuration](index.html#document-config){.reference
    .internal}
-   [Data Utilities](index.html#document-data_utils){.reference
    .internal}
-   [HyperParameter
    Search](index.html#document-hparam_search){.reference .internal}
-   [Camera Stream
    Predictions](index.html#document-predict_on_camera){.reference
    .internal}
-   [ASL Dataset](index.html#document-dataset){.reference .internal}
-   [Data Utilities](index.html#document-dl_utils){.reference .internal}
-   [Model Training](index.html#document-trainer){.reference .internal}
-   [Data Visualizations](index.html#document-visualizations){.reference
    .internal}
-   [Pytorch Models](index.html#document-pytorch_models){.reference
    .internal}
-   [Tensorflow
    Models](index.html#document-tensorflow_models){.reference .internal}
-   [Torch Lightning
    Models](index.html#document-lightning_models){.reference .internal}

::: {.relations}
### Related Topics

-   [Documentation overview](#)
:::
:::
:::

::: {.clearer}
:::
:::

::: {.footer}
©2023, Asad Bin Imtiaz, Felix Schlatter. \| Powered by [Sphinx
4.5.0](http://sphinx-doc.org/) & [Alabaster
0.7.12](https://github.com/bitprophet/alabaster)
:::
