import warnings

import numpy as np

# PySINDy が内部で pkg_resources を参照するため発生する既知の警告を抑制
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pysindy",
)

warnings.filterwarnings(
    "ignore",
    message="The attribute `n_input_features_` was deprecated",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    category=np.ComplexWarning,
)
