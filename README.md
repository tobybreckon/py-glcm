# Installation

py-glcm is a C++ extension, so you need a C++ compiler and the Python development headers:

* **Arch:** `sudo pacman -S base-devel`
* **Fedora:** `sudo dnf install gcc-c++ python3-devel`
* **Debian/Ubuntu:** `sudo apt install build-essential python3-dev`

Then, from the repository root (ideally inside a virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

`pip install .` builds the extension against NumPy 2.x in an isolated build environment. The resulting module also
works with NumPy >= 1.25 at runtime. Use `pip install . -v` to see the compiler output.

Do not use `python setup.py install`; it is deprecated by setuptools.

## Running the tests

Install the test dependencies, then run the tests as modules from the repository root:

```bash
pip install scipy scikit-image
python -m tests.sanity
python -m tests.features
```

Every line of output should report **check!**. After changing `py-glcm/core/src/glcm.cpp`, rerun `pip install .`
to rebuild before testing again.

`tests/benchmark.py` compares speed against other libraries and additionally requires `SimpleITK` and `pyradiomics`.

# Usage

* [Grey Level Co-occurrence Matrix](#grey-level-co-occurrence-matrix)
  * [glcm.glcm](#glcmglcm)
  * [glcm.xglcm](#glcmxglcm)
* [GLCM Features](#glcm-features)
  * [glcm.glcm_features](#glcmglcm_features)
  * [Supported Features](#supported-features)

```python
import numpy as np
import glcm

img = (np.random.rand(64, 64) * 8).astype(np.int32)

# One GLCM for distance 1, summed over directions N, NE, E and SE
m = glcm.glcm(img, [1], [1, 2, 3, 4], "sum", bins=8)

# Select features by combining flags with |
features = glcm.glcm_features(m, glcm.asm | glcm.contrast)
print(features["ASM"], features["Contrast"])
```

# Grey Level Co-occurrence Matrix

## glcm.glcm

```python
glcm.glcm(array, dists, dirs, mode, symmetric=True, bins=256, normalized=True, check=True)
```

Generates a GLCM.

### Parameters

* **array** : *array_like*

  Input image, either 2D or 3D.

* **dists** : *array_like*

  Array of desired integer distances `[d1, d2, ..., dn]`.

* **dirs** : *array_like*

  Array of desired integer directions `[d1, d2, ..., dn]`. 1 corresponds to N, 2 to NE, 3 to E, ... and 8 to NW.

  > **Caution!** Directions will be reordered in ascending order. To avoid confusing outputs, make sure to provide
  > an ordered array.

* **mode** : *string*

  Operation mode of the GLCM, either `"sum"` or `"raw"`.

  In raw mode, a GLCM is generated for every combination of distances and directions.

  In sum mode, all desired directions are added together, so only one GLCM per distance is generated. This is far
  more efficient than summing up afterwards.

* **symmetric** : *boolean, optional*

  > **Caution!** Running in symmetric mode will remove opposing directions and will replace directions 5 to 8 with
  > their corresponding counterparts. To avoid confusing outputs, only use directions 1 to 4 in symmetric mode.

* **bins** : *int, optional*

  Number of bins. When input checking is enabled, the input image is binned to this number if the maximum image
  value exceeds the number of bins. This also happens if the input image is not of integer type.

* **normalized** : *boolean, optional*

  Determines whether the output is normalized.

  > **Caution!** Normalization will fail if the input image has more than 10^15 pixels.

* **check** : *boolean, optional*

  Determines whether the input is checked for correctness.

### Returns

* **glcm** : *ndarray*

  Array of GLCM(s) with shape `[distances][directions][channels][bins][bins]`.

## glcm.xglcm

```python
glcm.xglcm(array, dists, dirs, mode, symmetric=True, bins=256, normalized=True, check=True)
```

Generates a GLCM for every channel combination.

### Parameters

* **array** : *array_like*

  Input image with three dimensions and shape `[dimx, dimy, channels]`.

* **dists** : *array_like*

  Array of desired integer distances `[d1, d2, ..., dn]`.

* **dirs** : *array_like*

  Array of desired integer directions `[d1, d2, ..., dn]`. 1 corresponds to N, 2 to NE, 3 to E, ... and 8 to NW.

  > **Caution!** Directions will be reordered in ascending order. To avoid confusing outputs, make sure to provide
  > an ordered array.

* **mode** : *string*

  Operation mode of the GLCM, either `"sum"` or `"raw"`.

  In raw mode, a GLCM is generated for every combination of distances and directions.

  In sum mode, all desired directions are added together, so only one GLCM per distance is generated. This is far
  more efficient than summing up afterwards.

* **symmetric** : *boolean, optional*

  > **Caution!** Running in symmetric mode will remove opposing directions and will replace directions 5 to 8 with
  > their corresponding counterparts. To avoid confusing outputs, only use directions 1 to 4 in symmetric mode.

* **bins** : *int, optional*

  Number of bins. When input checking is enabled, the input image is binned to this number if the maximum image
  value exceeds the number of bins. This also happens if the input image is not of integer type.

* **normalized** : *boolean, optional*

  Determines whether the output is normalized.

  > **Caution!** Normalization will fail if the input image has more than 10^15 pixels.

* **check** : *boolean, optional*

  Determines whether the input is checked for correctness.

### Returns

* **glcm** : *ndarray*

  Array of GLCM(s) with shape `[distances][directions][source channels][target channels][bins][bins]`.

# GLCM Features

## glcm.glcm_features

```python
glcm.glcm_features(array, features, symmetric=True, normalized=True)
```

Calculates features from a given set of GLCMs.

### Parameters

* **array** : *array_like*

  Array of GLCMs, where the last two axes hold the GLCM entries.

* **features** : *int*

  Bit flags selecting the desired features, combined with `|`. See [Supported Features](#supported-features).

* **symmetric** : *boolean, optional*

  Indicates whether the input GLCM(s) are symmetric.

* **normalized** : *boolean, optional*

  Indicates whether the input is normalized.

### Returns

* **features** : *dict*

  Dictionary mapping feature names to arrays of feature values.

## Supported Features

| Feature                   | Flag                | Dictionary key         | Status              |
|---------------------------|---------------------|------------------------|---------------------|
| Angular Second Moment     | `glcm.asm`          | `"ASM"`                |                     |
| Contrast                  | `glcm.contrast`     | `"Contrast"`           |                     |
| Correlation               | `glcm.correl`       | `"Correlation"`        | Not implemented yet |
| Autocorrelation           | `glcm.autocorrel`   | `"Autocorrelation"`    |                     |
| Sum of Squares            | `glcm.ssq`          | `"SSQ"`                | Not implemented yet |
| Inverse Difference Moment | `glcm.idm`          | `"IDM"`                |                     |
| Inverse Difference        | `glcm.idf`          | `"IDF"`                |                     |
| Sum Average               | `glcm.sumavg`       | `"Sum Average"`        |                     |
| Sum Variance              | `glcm.sumvar`       | `"Sum Variance"`       |                     |
| Sum Entropy               | `glcm.sumentrp`     | `"Sum Entropy"`        |                     |
| Difference Average        | `glcm.diffavg`      | `"Diff Average"`       |                     |
| Difference Variance       | `glcm.diffvar`      | `"Diff Variance"`      | Not implemented yet |
| Difference Entropy        | `glcm.diffentrp`    | `"Diff Entropy"`       |                     |
| Cluster Prominence        | `glcm.clusterprom`  | `"Cluster Prominence"` |                     |
| Cluster Shade             | `glcm.clustershade` | `"Cluster Shade"`      |                     |
| Cluster Tendency          | `glcm.clustertend`  | `"Cluster Tendency"`   |                     |
| Dissimilarity             | `glcm.dissim`       | `"Dissimilarity"`      |                     |

The Angular Second Moment, `ASM = sum over i, j of p(i, j)^2`, measures how homogeneous the patterns in the image are.
