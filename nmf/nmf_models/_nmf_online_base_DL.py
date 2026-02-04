import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ._nmf_base_DL import NMFBase_DL
from typing import Union

class NMFDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, torch.Tensor], chunk_size: int, dtype: torch.dtype):
        if isinstance(X, np.ndarray):
            self.X_cpu = torch.from_numpy(X).to(dtype=dtype)
        else:
            self.X_cpu = X.cpu().to(dtype=dtype)

        self.chunk_size = chunk_size
        self.n_samples = self.X_cpu.shape[0]
        self.n_chunks = (self.n_samples + chunk_size - 1) // chunk_size

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_samples)
        indices = torch.arange(start_idx, end_idx)
        return self.X_cpu[start_idx:end_idx], indices, start_idx



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
        max_pass: int = 20,
        chunk_size: int = 5000,
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

        self._max_pass = max_pass
        self._chunk_size = chunk_size
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
        i = 0
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors

        for chunk_data, chunk_indices, start_pos in self.dataloader:

            x = chunk_data.squeeze(0).to(self._device_type)
            idx = chunk_indices.squeeze(0) 
            h = self.H[idx, :]
            
            xWT = x @ self.W.T
            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)
            i += self._chunk_size

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def fit(self, X):
        super().fit(X)  

        # Create dataloader  
        dataset = NMFDataset(
            X=self.X,
            chunk_size=self._chunk_size,
            dtype=self._tensor_dtype
        )

        num_workers = max(0, self.n_jobs) if self.n_jobs != -1 else 0

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

        self._init_err = self._loss()   
        self._prev_err = self._init_err
