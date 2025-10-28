import torch
from typing import Union
from ._nmf_batch_base import NMFBatchBase

class NMFBatchCD(NMFBatchBase):
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
        max_iter: int = 500,
        hals_tol: float = 0.05,
        hals_max_iter: int = 200,
    ):
        assert beta_loss == 2.0, "Only supports Frobenius loss (β=2) for now."

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
            max_iter=max_iter,
        )

        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)
        self._hals_tol = hals_tol
        self._hals_max_iter = hals_max_iter


    # -------------------------
    # Update H given fixed W
    # -------------------------
    def _update_H(self):
        WWT = self.W @ self.W.T  # (k,k)
        XWT = self.X @ self.W.T  # (cell, k)

        eps = 1e-4 * torch.mean(torch.diag(WWT))
        I = torch.eye(WWT.shape[0], device=WWT.device)

        H_prev = self.H.clone()

        try:
            H_new = torch.linalg.solve(WWT + eps * I, XWT.T).T
        except RuntimeError:
            H_new = torch.linalg.lstsq((WWT + eps * I).T, XWT.T).solution.T


        H_new = torch.clamp(H_new, min=0.0, max=1e6)

        if torch.isnan(H_new).any():
            print("NaN detected in H — reverting to previous state")
            H_new = H_prev

        self.H = H_new


    # -------------------------
    # Update W given fixed H
    # -------------------------
    def _update_W(self):
    
        HTH = self.H.T @ self.H  # (k, k)
        XTH = self.X.T @ self.H # (gene, k)

        eps = 1e-4 * torch.mean(torch.diag(HTH))
        I = torch.eye(HTH.shape[0], device=HTH.device)

        W_prev = self.W.clone()

        # Solve (W^T W) H^T = (X^T W)
        try:
            W_new = torch.linalg.solve(HTH + eps * I, XTH.T).T
        except RuntimeError:
            W_new = torch.linalg.lstsq((HTH + eps * I).T, XTH.T).solution.T

        W_new = torch.clamp(W_new, min=0.0, max=1e6)

        # 🔍 Check for NaNs and revert if needed
        if torch.isnan(W_new).any():
            print("NaN detected in H — reverting to previous state")
            W_new = W_prev.T

        self.W = W_new.T


    # -------------------------
    # Main fit loop
    # -------------------------
    def fit(self, X):
        super().fit(X)

        for i in range(self._max_iter):
            self._update_H()
            torch.cuda.synchronize()

            self._update_W()
            torch.cuda.synchronize()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                print(f"Iter {i+1}: loss = {self._cur_err:.4e}")

                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"Converged after {self.num_iters} iterations.")
                    return
                self._prev_err = self._cur_err

        self.num_iters = self._max_iter
        print(f"Not converged after {self.num_iters} iterations.")