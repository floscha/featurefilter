# Featurefilter
[![Build Status](https://travis-ci.com/floscha/featurefilter.svg?branch=master)](https://travis-ci.com/floscha/featurefilter)
[![Coverage Status](https://coveralls.io/repos/github/floscha/featurefilter/badge.svg?branch=master)](https://coveralls.io/github/floscha/featurefilter?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/04e6164687e6456cbafdb09059e1d4e4)](https://www.codacy.com/app/floscha/featurefilter?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=floscha/featurefilter&amp;utm_campaign=Badge_Grade)
[![PyPI Version](https://img.shields.io/pypi/v/featurefilter.svg)](https://pypi.python.org/pypi/featurefilter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Featurefilter** is a Python library for removing uninformative variables from datasets.

## Features
- [x] 100% test coverage
- [x] Pandas backend
- [x] Support for scikit-learn pipelines
- [x] Support for scikit-learn selectors
- [ ] PySpark backend (planned for version 0.2)

## Usage Examples

All examples can also be found in the [example notebook](examples.ipynb).

### Remove columns with too many NA values
```python
import numpy as np
import pandas as pd

from featurefilter import NaFilter

df = pd.DataFrame({'A': [0, np.nan, np.nan],
                   'B': [0, 0, np.nan]})

na_filter = NaFilter(max_na_ratio=0.5)
na_filter.columns_to_drop = ['A']
na_filter.fit_transform(df)
```

### Remove columns with too low or high variance
```python
import pandas as pd

from featurefilter import VarianceFilter

df = pd.DataFrame({'A': [0., 1.], 'B': [0., 0.]})

variance_filter = VarianceFilter()
variance_filter.fit_transform(df)
```

### Remove columns with too high correlation to the target variables
```python
import pandas as pd

from featurefilter import TargetCorrelationFilter

df = pd.DataFrame({'A': [0, 0], 'B': [0, 1], 'Y': [0, 1]})

target_correlation_filter = TargetCorrelationFilter(target_column='Y')
target_correlation_filter.fit_transform(df)
```

### Remove columns using generalized linear models (GLMs)
```python
import pandas as pd

from featurefilter import GLMFilter

df = pd.DataFrame({'A': [0, 0, 1, 1],
                   'B': [0, 1, 0, 1],
                   'Y': [0, 0, 1, 1]})

glm_filter = GLMFilter(target_column='Y', top_features=1)
glm_filter.fit_transform(df)
```

### Remove columns using tree-based models
```python
import pandas as pd

from featurefilter import TreeBasedFilter

df = pd.DataFrame({'A': [0, 0, 1, 1],
                   'B': [0, 1, 0, 1],
                   'Y': ['a', 'a', 'b', 'b']})

tree_based_filter = TreeBasedFilter(target_column='Y',
                                    categorical_target=True,
                                    top_features=1)
tree_based_filter.fit_transform(df)
```

### Remove columns using multiple filters combined with scikit-learn's Pipeline API
```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from featurefilter import NaFilter, VarianceFilter

df = pd.DataFrame({'A': [0, np.nan, np.nan],
                   'B': [0, 0, 0],
                   'C': [0, np.nan, 1]})

pipeline = Pipeline([
    ('na_filter', NaFilter(max_na_ratio=0.5)),
    ('variance_filter', VarianceFilter())
])

pipeline.fit_transform(df)
```

### Remove columns using existing selectors provided by scikit-learn
```python
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from featurefilter import SklearnWrapper

df = pd.DataFrame({'A': [0, 0, 1, 1],
                   'B': [0, 1, 0, 1],
                   'Y': [0, 0, 1, 1]})

model = RFECV(LinearRegression(),
              min_features_to_select=1,
              cv=3)
selector = SklearnWrapper(model, target_column='Y')
selector.fit_transform(df)
```
