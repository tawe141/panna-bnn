from data.dataset import PANNADiamineDataset, InMemoryPANNA
import torch
import numpy as np
from utils.data import custom_collate
from data.transforms import BatchwiseStandardizeTransform, ElementwiseScalarStandardizeTransform


def test_getitem():
    ds = PANNADiamineDataset()
    sample = ds[0]
    assert isinstance(sample[0], torch.FloatTensor)
    assert isinstance(sample[1], torch.LongTensor)
    assert isinstance(sample[2], np.float64) or isinstance(sample[2], np.float32) or isinstance(sample[2], float)

    feature_length = sample[0].shape[1]
    print('Vector lengths: %i' % feature_length)


def test_transform():
    ds = PANNADiamineDataset(standardize_x=True, standardize_y=True)
    ds_nonstandard = PANNADiamineDataset()

    sample = ds[0]
    nonstandard_sample = ds_nonstandard[0]

    assert not torch.allclose(sample[0], nonstandard_sample[0])
    assert torch.allclose(sample[1], nonstandard_sample[1])
    assert not np.allclose(sample[2], nonstandard_sample[2])


def test_inmemory():
    ds = InMemoryPANNA(standardize_x=True, standardize_y=True)
    normal_ds = PANNADiamineDataset(standardize_x=True, standardize_y=True)

    sample = ds[0]
    normal_sample = normal_ds[0]

    assert torch.allclose(sample[0], normal_sample[0])
    assert torch.allclose(sample[1], normal_sample[1])
    assert np.allclose(sample[2], normal_sample[2])


def test_custom_collate():
    ds = PANNADiamineDataset()
    batch = [ds[i] for i in range(64)]
    custom_collate(batch)


def test_standardize():
    x = torch.rand(2, 10, 5)
    x = [x[0], x[1]]
    transform = BatchwiseStandardizeTransform()
    for i in x:
        transform.update(i)

    true_var, true_mean = torch.var_mean(torch.cat(x), dim=0, keepdim=True, unbiased=False)
    assert torch.allclose(transform.mean, true_mean)
    assert torch.allclose(transform.var, true_var)


def test_standardize_one():
    x = torch.rand(10, 5)
    transform = BatchwiseStandardizeTransform()
    transform.update(x)

    true_var, true_mean = torch.var_mean(x, dim=0, keepdim=True, unbiased=False)
    assert torch.allclose(transform.mean, true_mean)
    assert torch.allclose(transform.var, true_var)


def test_standardize_scalars():
    a = np.random.rand(10)

    true_mean = np.mean(a)
    true_var = np.var(a)

    transform = ElementwiseScalarStandardizeTransform()
    for i in a:
        transform.update(i)

    assert np.isclose(transform.mean, true_mean)
    assert np.isclose(transform.var, true_var)
