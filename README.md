# algorithm-snippets

Small, self-contained Python implementations of classic algorithms from different fields — each one a minimal experiment that shows how the algorithm works, not a packaged library.

## Snippets

### Computer Vision (`CV/`)

- **`fast-bilateral-filter/`** — the bilateral filter recast as a linear convolution in a *3D bilateral grid* (space × intensity), following [A Fast Approximation of the Bilateral Filter using a Signal Processing Approach (ECCV 2006)](https://link.springer.com/chapter/10.1007/11744085_44) by Paris and Durand. Includes the fast variant that pre-blurs and downsamples the grid before the Gaussian, then upsamples the result.
- **`harris-detector/`** — Harris corner detection from scratch: Sobel derivatives, the Gaussian-smoothed structure tensor, and the `det(M) − k·trace(M)²` corner response, with detected corners drawn on the image.

Both scripts run on the bundled `CV/lenna.jpg`.

### Math (`MATH/`)

- **`conjugate-gradient/`** — the conjugate gradient method solving a random 9×9 symmetric system, printing the residual norm per iteration to show convergence in at most n steps.

### Probability (`PROBABILITY/`)

- **`MRF/`** — binary image denoising with an Ising-model Markov random field, optimized by Iterated Conditional Modes (greedy single-pixel flips), as in PRML §8.3.3. Binarizes Lenna, adds salt noise, and recovers the clean image.

## Running

Each snippet is a standalone script; dependencies are just NumPy, SciPy, and OpenCV (`opencv-python`).

```bash
cd CV/harris-detector && python main.py
```

The MRF script expects to be run from the repository root (`python PROBABILITY/MRF/image_denoise_ICM.py`) since it loads `./CV/lenna.jpg`.

## License

MIT — see `LICENSE`.
