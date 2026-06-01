from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class TrafficBatchDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        size: int,
        pad_with_last_sample: bool = True,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.batch_size = int(batch_size)
        self.original_size = int(size)
        self.current_ind = 0
        self.pad_with_last_sample = bool(pad_with_last_sample)
        self.num_padding = (self.batch_size - (self.original_size % self.batch_size)) % self.batch_size if pad_with_last_sample else 0
        self.size = self.original_size + self.num_padding
        self.num_batch = int(self.size // self.batch_size)
        self.indices = np.arange(self.original_size, dtype=np.int64)
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.RandomState(self.seed) if self.seed is not None else None
        if shuffle:
            self.shuffle()

    def __len__(self):
        return self.num_batch

    def shuffle(self):
        if self.original_size > 1:
            if self.rng is None:
                self.indices = np.random.permutation(self.original_size)
            else:
                self.indices = self.rng.permutation(self.original_size)
        self.current_ind = 0

    def _materialize_batch_indices(self, start_ind: int, end_ind: int) -> np.ndarray:
        count = end_ind - start_ind
        batch_indices = np.empty(count, dtype=np.int64)
        limit = min(end_ind, self.original_size)
        valid_count = max(limit - start_ind, 0)
        if valid_count > 0:
            batch_indices[:valid_count] = self.indices[start_ind:limit]
        if valid_count < count:
            fill_value = self.indices[-1] if self.original_size > 0 else 0
            batch_indices[valid_count:] = fill_value
        return batch_indices

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                batch_indices = self._materialize_batch_indices(start_ind, end_ind)
                yield self._fetch_batch(batch_indices)
                self.current_ind += 1

        return _wrapper()

    def _fetch_batch(self, batch_indices: np.ndarray):
        raise NotImplementedError


class ArrayTrafficDataset(TrafficBatchDataset):
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        batch_size: int,
        pad_with_last_sample: bool = True,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.xs = xs
        self.ys = ys
        super().__init__(
            batch_size=batch_size,
            size=len(xs),
            pad_with_last_sample=pad_with_last_sample,
            shuffle=shuffle,
            seed=seed,
        )

    def _fetch_batch(self, batch_indices: np.ndarray):
        return self.xs[batch_indices, ...], self.ys[batch_indices, ...]


class WindowedSeriesDataset(TrafficBatchDataset):
    def __init__(
        self,
        series: np.ndarray,
        window_start: int,
        window_end: int,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
        batch_size: int,
        scaler,
        pad_with_last_sample: bool = True,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.series = series
        self.window_start = int(window_start)
        self.window_end = int(window_end)
        self.x_offsets = np.asarray(x_offsets, dtype=np.int64)
        self.y_offsets = np.asarray(y_offsets, dtype=np.int64)
        self.scaler = scaler
        super().__init__(
            batch_size=batch_size,
            size=max(self.window_end - self.window_start, 0),
            pad_with_last_sample=pad_with_last_sample,
            shuffle=shuffle,
            seed=seed,
        )

    def _fetch_batch(self, batch_indices: np.ndarray):
        starts = batch_indices + self.window_start
        x_positions = starts[:, None] + self.x_offsets[None, :]
        y_positions = starts[:, None] + self.y_offsets[None, :]
        x = np.asarray(self.series[x_positions, ...], dtype=np.float32)
        y = np.asarray(self.series[y_positions, ...], dtype=np.float32)
        x[..., 0] = self.scaler.transform(x[..., 0])
        y[..., 0] = self.scaler.transform(y[..., 0])
        return x, y
