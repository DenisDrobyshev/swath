"""swath — semantic segmentation of aerial and satellite imagery.

The package covers the full path a segmentation model takes in practice:
preparing tiles from a raster dataset, training a U-Net, running seamless
inference over rasters far larger than the GPU, and serving the result as a
georeferenced mask through a small web application.
"""

__version__ = "0.1.0"

from swath.tasks import TASKS, Task, get_task

__all__ = ["TASKS", "Task", "__version__", "get_task"]
