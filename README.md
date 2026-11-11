# Analytical Tomographic Image Reconstruction

> **NOTE:** I worked on this project a while ago during my coursework at TU Munich, reimplementing algorithms from Jeffrey A. Fessler's *Image Reconstruction: Algorithms and Analysis*. There were some bugs that made the results of some methods incorrect, and I could not figure out why (turns out it was small mix-ups in array axes, grid offsets and coordinate conventions). After a few years, I remembered this project and asked Claude to fix the bugs for me. So credit to Claude ;)

Welcome to the Analytical Tomographic Image Reconstruction repository! This collection of Python code provides implementations for various analytical tomographic image reconstruction algorithms. These algorithms play a crucial role in transforming projection data into detailed images, making them valuable tools in medical imaging, materials science, and more. Included are **Fourier Gridding**, **Backproject Filter**, **Filtered Backprojection** and **Convolve Backproject**.

![Reconstructions of the Shepp-Logan phantom with each method](images/overview.png)

## Sample Results

All images below are produced from the 256×256 Shepp-Logan phantom with 180 projections (0°–179°), using the code in this repository. See [tutorial.ipynb](tutorial.ipynb) for the full walkthrough.

### Radon Transform

Each column of the sinogram is one projection: the line integrals of the object along parallel rays at angle θ.

![Phantom and its sinogram](images/radon.png)

### Fourier Gridding

By the Fourier slice theorem, the 1-D Fourier transform of a projection at angle θ is a radial line through the 2-D Fourier transform of the object. The slices are interpolated from the polar grid onto a Cartesian grid, and an inverse 2-D FFT gives the image. The projections are zero-padded 4× first (oversampling), which keeps the interpolation error small.

![Fourier gridding: 1-D FT of projections, interpolated 2-D spectrum, reconstruction](images/fourier_gridding.png)

### Backprojection

Smearing every projection back across the image plane. With few angles the streaks are obvious; with all 180 angles the result is the object blurred by 1/r.

![Unfiltered backprojection with 1, 2, 10, 45 and 180 projections](images/backprojection.png)

### Backproject Filter

The 1/r blur of the backprojection is removed afterwards in 2-D: multiply its spectrum by the cone filter |ρ| and transform back. The backprojection is computed on a 2× larger grid, since the blur extends far outside the object.

![Backproject-filter: backprojection, spectrum before and after the cone filter, result](images/backproject_filter.png)

### Filtered Backprojection

Each projection is filtered with a 1-D ramp filter |ω| (optionally apodized) before backprojection. Smoother windows trade resolution for less noise.

![Filter responses, filtered sinogram, and FBP with each filter](images/filtered_backprojection.png)

### Convolve Backproject

The same filtering as FBP, but done in the spatial domain by convolving each projection with the inverse Fourier transform of the filter. The result is identical to FBP up to floating-point error.

![Spatial kernels, convolved sinogram, CBP reconstruction](images/convolve_backproject.png)

## Code Structure

1. **backprojection.py**
   - Implementation of the (filtered) Backprojection algorithm.

2. **data_generator.py**
   - Generates 2D data for testing purposes.
   - Options include loading an image, using the Shepp-Logan phantom, or generating a simple Disk object.

3. **filters.py**
   - Implementation of various filters used in tomographic image reconstruction.
   - Includes the ramp-type filters for Filtered Backprojection, their spatial kernels for Convolve Backproject, and the cone filter for Backproject Filter.

4. **fourier_gridding.py**
   - Performs radial-to-cartesian interpolation on a 2D Fourier Transform.

5. **fourier_transform.py**
   - 1D Fourier Transform of each projection (with optional oversampling).

6. **radon_transform.py**
   - Implementation of the Radon Transform.

7. **tutorial.ipynb**
   - Jupyter notebook demonstrating the usage of all reconstruction methods with examples and visualizations.

## Usage

Install the dependencies:

```bash
pip install numpy scipy scikit-image imageio matplotlib
```

Then explore the provided [tutorial.ipynb](tutorial.ipynb) notebook (run it from the repository root so the files in `data/` are found) for comprehensive examples and visualizations of each reconstruction method. You can experiment with different data sources and filters to see how these algorithms perform under various conditions.

## Contributing

We welcome contributions to enhance and expand this repository. If you have improvements, additional algorithms, or new features to suggest, please feel free to open an issue or submit a pull request.

## Acknowledgments

Special thanks to me for writing this and Prof. Lasser for the wonderful Seminar course

## License

This code is released under the MIT License. Feel free to use, modify, and distribute the code in accordance with the terms of the license.

Happy reconstructing! 🌐🖼️
