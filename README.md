Data-Bias-Reduction
===

It is a method to preprocess the training data, producing an adjusted dataset that is independent of the group variable with minimum information loss from the [paper](https://arxiv.org/abs/1810.08255) in the references.

The problem of preprocessing the data to ensure orthogonality to a group of variables with minimum information loss can be expressed as a minimization of the Frobenius distance between the original data and the approximated version, under the constraint the inner product between the reconstructed data matrix and the group variable's matrix. Given the particular structure of assumption for the reconstructed data matrix, this leads to the following optimization problem:

![optimization-problem](https://github.com/meowoodie/Avoiding-Bias-Data/blob/master/imgs/optimizatio-problem.png)

This problem can be solved by *sparse orthogonal to subgroup algorithm* shown below:

![algorithm](https://github.com/meowoodie/Avoiding-Bias-Data/blob/master/imgs/algorithm.png)

In this repository, `orthogen.py` implements an Python class for reducing the bias in raw data by generating the reconstructed data matrix. In addition, `orthocheck.R` and `test_realdata.py` have various of helper functions for evaluating and testing the framework on real crime reports provided by the Atlanta Police Department.

### Examples

```python
# raw data matrix
X = np.array([
    [1, 1, 0, 0.1, 0,   0],
    [1, 1, 0, 0,   0,   0],
    [0, 0, 1, 1,   0,   0],
    [0, 0, 1, 1,   0.1, 0],
    [0, 0, 0, 0,   1,   1]
])
# group variable
Z = np.array([
    [1,   1],
    [1,   1],
    [0,   0.1],
    [0.1, 0],
    [0,   0.1]
])
# rank
k = 3
# initiate an orthogonal data generator object
odg = OrthoDataGen(X, Z, k)
# train via sparse orthogonal to subgroup algorithm
odg.sog(t=1.414, tol=1e-2)
# print the reconstructed data matrix
print(odg.reconstruct())
```

```bash
[2018-11-23T13:18:48.881924-05:00] n = 5, p = 6, k = 3, X is 5 x 6, Z is 5 x 2.
[2018-11-23T13:18:48.882085-05:00] updating 1/3 ...
[2018-11-23T13:18:48.882739-05:00] ---------------------------------
[2018-11-23T13:18:48.882839-05:00] iter 0
[2018-11-23T13:18:48.882950-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.883081-05:00]	s_j change is 1.218, u_j change is 1.245
[2018-11-23T13:18:48.883217-05:00]	Frobenius measure is 3.675
[2018-11-23T13:18:48.883476-05:00] ---------------------------------
[2018-11-23T13:18:48.883568-05:00] iter 1
[2018-11-23T13:18:48.883673-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.883795-05:00]	s_j change is 0.589, u_j change is 0.007
[2018-11-23T13:18:48.883927-05:00]	Frobenius measure is 3.522
[2018-11-23T13:18:48.884177-05:00] ---------------------------------
[2018-11-23T13:18:48.884268-05:00] iter 2
[2018-11-23T13:18:48.884372-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.884492-05:00]	s_j change is 0.000, u_j change is 0.000
[2018-11-23T13:18:48.884629-05:00]	Frobenius measure is 3.522
[2018-11-23T13:18:48.884783-05:00] updating 2/3 ...
[2018-11-23T13:18:48.885068-05:00] ---------------------------------
[2018-11-23T13:18:48.885160-05:00] iter 0
[2018-11-23T13:18:48.885265-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.885387-05:00]	s_j change is 0.700, u_j change is 1.068
[2018-11-23T13:18:48.885519-05:00]	Frobenius measure is 3.132
[2018-11-23T13:18:48.885773-05:00] ---------------------------------
[2018-11-23T13:18:48.885864-05:00] iter 1
[2018-11-23T13:18:48.886045-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.886221-05:00]	s_j change is 0.633, u_j change is 0.009
[2018-11-23T13:18:48.886433-05:00]	Frobenius measure is 3.015
[2018-11-23T13:18:48.886763-05:00] ---------------------------------
[2018-11-23T13:18:48.886913-05:00] iter 2
[2018-11-23T13:18:48.887056-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.887311-05:00]	s_j change is 0.000, u_j change is 0.000
[2018-11-23T13:18:48.887841-05:00]	Frobenius measure is 3.015
[2018-11-23T13:18:48.888504-05:00] updating 3/3 ...
[2018-11-23T13:18:48.889314-05:00] ---------------------------------
[2018-11-23T13:18:48.889552-05:00] iter 0
[2018-11-23T13:18:48.889666-05:00]	||u_j||_1 = 1.414
[2018-11-23T13:18:48.889818-05:00]	s_j change is 2.191, u_j change is 2.704
[2018-11-23T13:18:48.889972-05:00]	Frobenius measure is 2.654
[2018-11-23T13:18:48.890345-05:00] ---------------------------------
[2018-11-23T13:18:48.890503-05:00] iter 1
[2018-11-23T13:18:48.890650-05:00]	||u_j||_1 = 0.000
[2018-11-23T13:18:48.890820-05:00]	s_j change is 1.979, u_j change is 1.000
[2018-11-23T13:18:48.891015-05:00]	Frobenius measure is 2.457
[2018-11-23T13:18:48.891360-05:00] ---------------------------------
[2018-11-23T13:18:48.891466-05:00] iter 2
[2018-11-23T13:18:48.891713-05:00]	||u_j||_1 = 0.000
[2018-11-23T13:18:48.892035-05:00]	s_j change is 1.000, u_j change is 0.000
[2018-11-23T13:18:48.892368-05:00]	Frobenius measure is 2.457
[2018-11-23T13:18:48.893334-05:00] ---------------------------------
[2018-11-23T13:18:48.893529-05:00] iter 3
[2018-11-23T13:18:48.893684-05:00]	||u_j||_1 = 0.000
[2018-11-23T13:18:48.893887-05:00]	s_j change is 0.000, u_j change is 0.000
[2018-11-23T13:18:48.894038-05:00]	Frobenius measure is 2.457
[[ 0.          0.         -0.04971556 -0.04950346  0.          0.        ]
 [ 0.          0.         -0.14971298 -0.14907425  0.          0.        ]
 [ 0.          0.          1.99927592  1.99074618  0.          0.        ]
 [ 0.          0.          1.99428549  1.98577704  0.          0.        ]
 [ 0.          0.         -0.00499043 -0.00496914  0.          0.        ]]
```

### Real data

```python
# keywords that we are studying
X_KEYWORDS   = ['stole', 'robbery', 'males']
Z_KEYWORDS   = ['black', 'black_males']

# load raw data matrix X (exclude biased keywords) and subgroup variable Z (biased keywords)
_, XZ = utils.extract_keywords_from_corpus(keywords=all_keyword)

# configurations
all_keyword   = X_KEYWORDS + Z_KEYWORDS
x_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in X_KEYWORDS ]
z_col_idx     = [ ALL_KEYWORDS.index(keyword) for keyword in Z_KEYWORDS ]
valid_row_idx = np.where(XZ.sum(axis=1) > 0)[0]

# remove rows with all zero entries
XZ = XZ[valid_row_idx, :]
print('[%s] raw data matrix is %d x %d' % (arrow.now(), XZ.shape[0], XZ.shape[1]), file=sys.stderr)
# split XZ into X and Z according to their keywords
X = XZ[:, x_col_idx]
Z = XZ[:, z_col_idx]
# initiate orthogonal data generator
odg = orthogen.OrthoDataGen(X, Z, k=2)
odg.sog(t=1.414, tol=1e-2)
# X_hat = odg.reconstruct()
```

```bash
Dictionary(14281 unique tokens: ['use_cell', 'raining', 'ward_unit', 'group_males', 'call_regards']...)
MmCorpus(10056 documents, 14281 features, 1016964 non-zero entries)
extraced keywords ids: [1056, 8633, 8011, 9072, 13018]
(3025, 5)
[2018-11-23T14:25:11.580836-05:00] n = 3025, p = 3, k = 2, X is 3025 x 3, Z is 3025 x 2.
[2018-11-23T14:25:11.581021-05:00] updating 1/2 ...
[2018-11-23T14:25:16.807641-05:00] ---------------------------------
[2018-11-23T14:25:16.807786-05:00] iter 0
[2018-11-23T14:25:16.807891-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:16.808038-05:00]	s_j change is 31.264, u_j change is 0.809
[2018-11-23T14:25:16.808195-05:00]	Frobenius measure is 37.196
[2018-11-23T14:25:16.861449-05:00] ---------------------------------
[2018-11-23T14:25:16.861593-05:00] iter 1
[2018-11-23T14:25:16.861720-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:16.861869-05:00]	s_j change is 0.558, u_j change is 0.000
[2018-11-23T14:25:16.862027-05:00]	Frobenius measure is 37.174
[2018-11-23T14:25:16.904917-05:00] ---------------------------------
[2018-11-23T14:25:16.905060-05:00] iter 2
[2018-11-23T14:25:16.905167-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:16.905317-05:00]	s_j change is 0.000, u_j change is 0.000
[2018-11-23T14:25:16.905464-05:00]	Frobenius measure is 37.174
[2018-11-23T14:25:16.905680-05:00] updating 2/2 ...
[2018-11-23T14:25:22.033835-05:00] ---------------------------------
[2018-11-23T14:25:22.033988-05:00] iter 0
[2018-11-23T14:25:22.034124-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:22.034260-05:00]	s_j change is 31.402, u_j change is 1.006
[2018-11-23T14:25:22.034417-05:00]	Frobenius measure is 2.768
[2018-11-23T14:25:22.086285-05:00] ---------------------------------
[2018-11-23T14:25:22.086430-05:00] iter 1
[2018-11-23T14:25:22.086558-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:22.086692-05:00]	s_j change is 0.672, u_j change is 0.000
[2018-11-23T14:25:22.086847-05:00]	Frobenius measure is 2.659
[2018-11-23T14:25:22.129671-05:00] ---------------------------------
[2018-11-23T14:25:22.129830-05:00] iter 2
[2018-11-23T14:25:22.129968-05:00]	||u_j||_1 = 1.000
[2018-11-23T14:25:22.130107-05:00]	s_j change is 0.000, u_j change is 0.000
[2018-11-23T14:25:22.130275-05:00]	Frobenius measure is 2.659
```

### Results

![results](https://github.com/meowoodie/Avoiding-Bias-Data/blob/master/imgs/results.png)

### References

- [Emanuele Aliverti, Kristian Lum, James E. Johndrow, David B. Dunson. "Removing the influence of a group variable in high-dimensional predictive modelling"](https://arxiv.org/abs/1810.08255)
- [S. Zhu and Y. Xie. "Crime Linkage Detection by Spatio-Temporal Text Point Processes"](https://arxiv.org/abs/1902.00440)
