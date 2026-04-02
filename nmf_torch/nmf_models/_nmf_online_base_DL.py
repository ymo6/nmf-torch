import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ._nmf_base_DL import NMFBase_DL
from typing import Union

class NMFDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, torch.Tensor], dtype: torch.dtype):
        if isinstance(X, np.ndarray):
            self.X_cpu = torch.from_numpy(X).to(dtype=dtype)
        else:
            self.X_cpu = X.cpu().to(dtype=dtype)
        self.n_samples = self.X_cpu.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X_cpu[idx], idx



class NMFOnlineBase_DL(NMFBase_DL):
    def __init__(
        self,
        n_components: int,
        init: str,
        beta_loss: float,
        tol: float,
        random_state: int,
        alpha_W: float,
        l1_ratio_W: float,
        alpha_H: float,
        l1_ratio_H: float,
        fp_precision: Union[str, torch.dtype],
        device_type: str,
        n_jobs: int = -1,
        max_epoch: int = 20,
        minibatch_size: int = 5000,
        shuffle: bool = True,
    ):
        assert beta_loss == 2.0 # only work for F norm for now

        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            random_state=random_state,
            alpha_W=alpha_W,
            l1_ratio_W=l1_ratio_W,
            alpha_H=alpha_H,
            l1_ratio_H=l1_ratio_H,
            fp_precision=fp_precision,
            device_type=device_type,
            n_jobs=n_jobs,
        )

        self._max_epoch = max_epoch
        self._minibatch_size = minibatch_size
        self._shuffle = shuffle
        self.n_jobs = n_jobs


    def _h_err(self, h, hth, WWT, xWT):
        # Forbenious-norm^2 in trace format (No X)
        res = self._trace(WWT, hth) / 2.0 - self._trace(h, xWT)
        # Add regularization terms if needed
        if self._l1_reg_H > 0.0:
            res += self._l1_reg_H * h.norm(p=1)
        if self._l2_reg_H > 0.0:
            res += self._l2_reg_H * torch.trace(hth) / 2.0
        return res

    def _loss(self):
        """ calculate loss online by passing through all data"""
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors

        for x, idx in self.dataloader:
            x = x.to(self._device_type)
            h = self.H[idx, :]

            xWT = x @ self.W.T
            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def fit(self, X):
        super().fit(X)

        # Create dataloader
        dataset = NMFDataset(
            X=self.X,
            dtype=self._tensor_dtype
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self._minibatch_size,
            shuffle=self._shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        self._init_err = self._loss()
        self._prev_err = self._init_err
