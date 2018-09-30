import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.

    Args
        label: The label to get the color for.

    Returns
        A list of three values representing a RGB color.

        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


"""
Generated using:

```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
shuffle(colors)
pprint(colors)
```
"""

colors = [[127, 0, 255],
          [0, 255, 255],
          [255, 255, 0],
          [255, 127, 0],
          [0, 0, 255],
          [255, 0, 127],
          [0, 255, 127],
          [0, 255, 0],
          [0, 127, 255],
          [255, 0, 255],
          [127, 255, 0],
          [255, 0, 0]]

if __name__ == '__main__':
    import numpy as np
    import matplotlib.colors
    import pprint

    class_num = 12
    colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in
              np.arange(0, 1, 1.0 / class_num)]
    np.random.shuffle(colors)
    pprint.pprint(colors)
