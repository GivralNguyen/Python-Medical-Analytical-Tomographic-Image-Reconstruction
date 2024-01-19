This repository contains Python code for various analytical tomographic image reconstruction algorithms. The supported methods include Backprojection, Backproject Filter, Filter Backprojection, Fourier Gridding, and Convolve Backprojection.

Code Structure
1. backprojection.py:
Implementation of the Backprojection algorithm.
2. data_generator.py:
Generates 2D data for testing purposes.
Options include loading an image, using the Shepp-Logan phantom, or generating a simple Disk object.
3. filters.py:
Implementation of various filters used in tomographic image reconstruction.
Includes filters for Backproject Filter and Filter Backprojection algorithms.
4. fourier_gridding.py:
Performs radial-to-cartesian interpolation on a 2D Fourier Transform.
5. fourier_transform.py:
Implementation of the Fourier Transform.
6. radon_transform.py:
Implementation of the Radon Transform.
7. tutorials.ipynb:
Jupyter notebook demonstrating the usage of all reconstruction methods with examples and visualizations.
