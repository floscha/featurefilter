import unittest

import pytest

from featurefilter.abstract_transformer import AbstractTransformer


def test_fit_raises_not_implemented_error():
    test_transformer = AbstractTransformer()

    with pytest.raises(NotImplementedError):
        test_transformer.fit(None)


def test_transform_raises_not_implemented_error():
    test_transformer = AbstractTransformer()

    with pytest.raises(NotImplementedError):
        test_transformer.transform(None)


def test_fit_transform_raises_not_implemented_error():
    test_transformer = AbstractTransformer()

    with pytest.raises(NotImplementedError):
        test_transformer.fit_transform(None)


if __name__ == '__main__':
    unittest.main()
