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
            max_iter=max_iter,
        )
        print("successfully enter cd class")


        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)
        self._hals_tol = hals_tol
        self._hals_max_iter = hals_max_iter


    def _update_H(self):
        #for i in range(self._hals_max_iter):
            #cur_max = 0.0    
        for k in range(self.k):
            for j in range(self.H.shape[0]):  # Update one parameter at a time
                # Compute gradient: grad = (H @ W.T @ W)[j, k] - (X @ W.T)[j, k]
                grad = -self._XWT[j, k]
                for r in range(self.k):
                    grad += self._WWT[k, r] * self.H[j, r]
                
                # Add regularization to gradient
                if self._l1_reg_H > 0.0:
                    grad += self._l1_reg_H
                
                # Compute Hessian (diagonal element)
                hess = self._WWT[k, k]
                if self._l2_reg_H > 0.0:
                    hess += self._l2_reg_H
                
                # Update parameter
                h_old = self.H[j, k]
                if hess != 0:
                    h_new = max(h_old - grad / hess, 0.0)
                else:
                    h_new = 0.0
                
                #cur_max = max(cur_max, torch.abs(h_old - h_new))
                self.H[j, k] = h_new
            
            #print(f" niter={i+1}, loss={cur_max / self.H.mean()}.")

            #if i + 1 < self._hals_max_iter and cur_max / self.H.mean() < self._hals_tol:
            #    break            

        self._HTH = self.H.T @ self.H


    def _update_W(self):
        HTX = self.H.T @ self.X
        #for i in range(self._hals_max_iter):
        #    cur_max = 0.0
        for k in range(self.k):
            for j in range(self.W.shape[1]):  # Update one parameter at a time
                # Compute gradient: grad = (H.T @ H @ W)[k, j] - (H.T @ X)[k, j]
                grad = -HTX[k, j]
                for r in range(self.k):
                    grad += self._HTH[k, r] * self.W[r, j]
                
                # Add regularization to gradient
                if self._l1_reg_W > 0.0:
                    grad += self._l1_reg_W
                
                # Compute Hessian (diagonal element)
                hess = self._HTH[k, k]
                if self._l2_reg_W > 0.0:
                    hess += self._l2_reg_W
                
                # Update parameter
                w_old = self.W[k, j]
                if hess != 0:
                    w_new = max(w_old - grad / hess, 0.0)
                else:
                    w_new = 0.0

                self.W[k, j] = w_new
         #          cur_max = max(cur_max, torch.abs(w_old - w_new))
                

         #   if i + 1 < self._hals_max_iter and cur_max / self.W.mean() < self._hals_tol:
         #      break

        self._WWT = self.W @ self.W.T
        self._XWT = self.X @ self.W.T


    def fit(self, X):
        super().fit(X)

        # Batch update.
        for i in range(self._max_iter):
            self._update_H()
            self._update_W()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                print(f" niter={i+1}, loss={self._cur_err}.")
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    return

                self._prev_err = self._cur_err

        self.num_iters = self._max_iter
        print(f"    Not converged after {self.num_iters} iteration(s).")
