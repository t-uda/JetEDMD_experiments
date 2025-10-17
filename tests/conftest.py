import warnings

# PySINDy が内部で pkg_resources を参照するため発生する既知の警告を抑制
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pysindy",
)
