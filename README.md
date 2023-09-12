# crispyn
CRIteria Significance determining in PYthoN - The Python 3 Library for determining criteria weights for MCDA methods.

This library provides 15 criteria weighting methods: 11 objective, 4 subjective and a Stochastic Multicriteria Acceptability Analysis Method (SMAA) 
that does not require criteria weights.

## Installation

```
pip install crispyn
```

## Usage

`crispyn` is the Python 3 package that provides 15 weighting methods: 11 objective and 4 subjective, which can be used to determine criteria weights for 
solving multi-criteria problems with Multi-Criteria Decision Analysis (MCDA) methods. The first step is providing the decision matrix `matrix` with alternatives' 
performance values. The decision matrix is two-dimensional and contains m alternatives in rows and n criteria in columns. You also have to provide 
criteria types `types`. Criteria types are equal to 1 for profit criteria and -1 for cost criteria. Then you have to calculate criteria weights 
using the weighting method chosen from `crispyn.weighting_methods` submodule. Depending on the chosen objective method, you have to provide `matrix` or `matrix` and `types` as 
weighting method arguments. In the case of subjective weighting methods, provided parameters are different, such as ordered criteria indexes and significance values assigned by the decision-maker to criteria. It is detailed in Usage in the documentation. Then, you can evaluate alternatives from the decision matrix using the VIKOR method 
from `weighting method.mcda_methods` module. The VIKOR method returns a vector with preference values `pref` assigned to alternatives. To rank alternatives 
according to VIKOR preference values, you have to sort them in ascending order because, in the VIKOR method, the best alternative has the lowest 
preference value. The alternatives are ranked using the `rank_preferences` method provided in the `crispyn.additions` submodule. Parameter `reverse = False` means that alternatives 
are sorted in ascending order. Here is an example of using the Entropy weighting method `entropy_weighting` for determining criteria weights and 
the VIKOR method to calculate preference values:

```python
import numpy as np
from crispyn.mcda_methods import VIKOR
from crispyn import weighting_methods as mcda_weights
from crispyn import normalizations as norms
from crispyn.additions import rank_preferences

matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
[256, 8, 32, 1.0, 1.8, 6919.99],
[256, 8, 53, 1.6, 1.9, 8400],
[256, 8, 41, 1.0, 1.75, 6808.9],
[512, 8, 35, 1.6, 1.7, 8479.99],
[256, 4, 35, 1.6, 1.7, 7499.99]])

types = np.array([1, 1, 1, 1, -1, -1])
weights = mcda_weights.entropy_weighting(matrix)

# Create the VIKOR method object
vikor = VIKOR(normalization_method=norms.minmax_normalization)
# Calculate alternatives preference function values with VIKOR method
pref = vikor(matrix, weights, types)
# Rank alternatives according to preference values
rank = rank_preferences(pref, reverse = False)
```

### Stochastic Multicriteria Acceptability Analysis Method (SMAA)

Additionally, the Crispyn library provides the Stochastic Multicriteria Acceptability Analysis Method (SMAA), which, combined 
with the VIKOR method, is designed to solve decision problems when there is a lack of information about criteria preferences (unknown criteria 
weights). This method is implemented in the class named `VIKOR_SMAA`. This method requires only the decision matrix, a matrix with 
weight vectors and criteria types provided in one call. The number of weight vectors is equal to the number of iterations. First, the matrix with
weight vectors must be generated with `_generate_weights` method provided by the `VIKOR_SMAA` class. In this method, uniform distributed weights 
are generated by Monte Carlo simulation. The results of the provided `VIKOR_SMAA` method are Rank acceptability index, Central weight vector, and 
Rank scores.

### Rank acceptability index

The ranking is built based on generated weights. Next, counters for corresponding ranks in relation to the alternatives are increased. 
After a given number of iterations, the rank acceptability indexes are obtained by dividing the counters by the number of iterations. 
Rank acceptability index shows the share of different scores placing an alternative in a given rank. 

### Central weight vector

The central weights are calculated similarly. In each iteration, the weight vector is added to its ‘summed weight vector’ when the 
alternative gets the rank. Next, this vector is divided by the number of iterations to get the central weight vector. The central weight 
vector describes the preferences of a typical decision-maker, supporting this alternative with the assumed preference model. It allows the 
decision-maker to see what criteria preferences result in the best evaluation of given alternatives.

### Rank scores

Final ranking of alternatives provided by the ranking function, which adds to each alternative value of 1 each time it has better preference 
values than each other.

Here is example of use of the `VIKOR_SMAA` method:

```python
from crispyn.mcda_methods import VIKOR_SMAA

# criteria number
n = matrix.shape[1]
# SMAA iterations number
iterations = 10000

# create the VIKOR_SMAA method object
vikor_smaa = VIKOR_SMAA()

# generate multiple weight vectors in matrix
weight_vectors = vikor_smaa._generate_weights(n, iterations)

# run the vikor_smaa method
rank_acceptability_index, central_weight_vector, rank_scores = vikor_smaa(matrix, weight_vectors, types)
```

## License

`crispyn` was created by Aleksandra Bączkiewicz. It is licensed under the terms of the MIT license.

## Documentation

Documentation of this library with instruction for installation and usage is 
provided [here](https://crispyn.readthedocs.io/en/latest/)
