# Python 101
### Python distributions
  - [official distribution](https://www.python.org) at www.python.org
  - [Anaconda](https://www.continuum.io/downloads) from Continuum Analytics
  - [IPython](http://ipython.org/)
  A more detailed list can be found [here](https://wiki.python.org/moin/PythonDistributions).

### Working environment
Create a virtual environment:
```sh
$ conda create --name ml python=3.5 
```
Reference: https://conda.io/docs/using/envs.html#create-a-separate-environment

### Basic packages for Machine Learning
 - numpy
 - matplotlib
 - scipy

##### Package installation
  - `pip`
  - `conda`
    Example:
    ```sh
    $ pip install numpy
    $ conda install matplotlib
    ```
    References:
    [1] https://conda.io/docs/using/pkgs.html#install-a-package
    

## Linear Regression

### Problem Statement
  Assume that the system can be modeled as

  $$ y \approx f(\mathbf{x}) = \widehat{y} $$

  $$ f(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_M x_M $$

where
  - input $$\mathbf{x} = [x_{1}, x_{2}, ..., x_{M}]^T$$
  - output $$y$$
  
Given $$ N $$ inputs $$\mathbf{x}_1, ..., \mathbf{x}_N$$ and corresponding outputs $$ y_1, ..., y_N $$, estimate the coefficients $$ \mathbf{w} = [w_0, w_1, ..., w_{M}]^T $$

### Solution
For an input $$ \mathbf{x}_i $$, let $$ \bar{\mathbf{x}}_i = [1, x_{i1}, ..., x_{iM}]^T $$ be the extended input

The model become

$$ y_i \approx \bar{\mathbf{x}}_i^T \mathbf{w} = \widehat{y}_i $$

Prediction error:

$$ \frac{1}{2} e_i^2 = \frac{1}{2} (y_i - \widehat{y}_i)^2 = \frac{1}{2} (y_i - \bar{\mathbf{x}}_i^T \mathbf{w})^2 $$

Cost function for :

$$ \mathcal{L}(\mathbf{w}) = \frac{1}{2} \sum_{i = 1}^{N} (y_i - \bar{\mathbf{x}_i}^T \mathbf{w})^2 = \frac{1}{2} || \mathbf{y} - \bar{\mathbf{X}} \mathbf{w} ||_2^2$$

where
 - $$ \bar{\mathbf{X}} = [\bar{\mathbf{x}}_1, \bar{\mathbf{x}}_2 ..., \bar{\mathbf{x}}_N] $$
 - $$ \mathbf{y} = [y_1, y_2, ..., y_N]^T $$
 - $$ || . ||_2 $$ is Euclidean norm


Optimal solution:

$$ \mathbf{w}^{*} = \mathrm{argmin}\limit_{\mathbf{w}} \mathcal{L}(\mathbf{w}) 
$$

Solve for $$ \mathbf{w}^{*} $$ by setting the gradient of $$ \mathcal{L}(\mathbf{w}) $$ to $$ \mathbf{0} $$

$$ \frac{\partial\mathcal{L}(\mathbf{w})}{\partial \mathbf{w}}
= \bar{\mathbf{X}}^T (\bar{\mathbf{X}} \mathbf{w} - \mathbf{y})
= \mathbf{0} $$

Solution:

$$ \mathbf{w}^{*} 
 = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^T \bar{\mathbf{X}}^T \mathbf{y} = \mathbf{A}^{\dagger} \mathbf{b}$$


### Solve with Python
In this section, we will
 - use the `numpy` module to represent vectors and matrices and perform computations to obtain the solution.
 - use the `matplotlib` module for plots

#### Example problem
Given data of heights and weights of 15 people
| Height        | Weight        |
|---------------|-------------|
| 147	        | 49	|
| 168	        | 60    |
| 150	        | 50	|
| 170	        | 72    |
| 153	        | 51	|
| 173	        | 63    |
| 155	        | 52	|
| 175	        | 64    |
| 158	        | 54	|
| 178	        | 66    |
| 160	        | 56	| 
| 180	        | 67    |
| 163	        | 58	|
| 183	        | 68    |
| 165	        | 59    |
Predict the weight of a person given the height.

#### Solution in Python with `numpy` 
Start with importing the required modules:

```py
import numpy as np
import matplotlib.pyplot as plt
```

Next, initialize and plot the given data.

Python has a built-in type `list` to represent an ordered collection of objects. For example,
```py
a = [1, 2, 7, 6 ]
b = [ [1, 2, 3], [3, 4, 5] ]
```
The type `list` does not support computations on vector and matrix, so we need to convert objects of type `list` to type `numpy.ndarray`.
```py
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
```
Note that `numpy.ndarray` represents arrays, not matrices (as in MATLAB). Here, a matrix is represented by a 2-D array.
```py
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![Fig. 1](http://machinelearningcoban.com/assets/LR/output_3_0.png)

Now, we build matrices and vectors required to solve for $$\mathbf{w}^{*}$$:

$$ \bar{\mathbf{X}} = [\bar{\mathbf{x}}_1, \bar{\mathbf{x}}_2 ..., \bar{\mathbf{x}}_N] $$

```py
# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
```

$$ \mathbf{A}^{\dagger}
 = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^T $$

```py
A = np.dot(Xbar.T, Xbar)
```

$$ \mathbf{b}
 = \bar{\mathbf{X}}^T \mathbf{y} $$

```py
b = np.dot(Xbar.T, y)
```

$$ \mathbf{w}^{*} 
 = \mathbf{A}^{\dagger} \mathbf{b}$$

```py
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
```

Visual illustration

```py
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![Fig. 2](http://machinelearningcoban.com/assets/LR/output_5_1.png)

#### Solution in Python with `scikit-learn` 

```py
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
```

