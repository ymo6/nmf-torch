import collections
import torch

from typing import Optional, Union
from ._nmf_online_base import NMFOnlineBase


class NMFOnlineHALS(NMFOnlineBase):
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
        minibatch_max_iter: int = 200,
        h_tol: float = 0.05,
        w_tol: float = 0.05,
        shuffle: bool = True,
        verbose: bool = False,
        rescale: bool = True,
        patience: Optional[int] = 10,
        tol_window: int = 10,
    ):
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
            max_epoch=max_epoch,
            minibatch_size=minibatch_size,
        )

        self._minibatch_max_iter = minibatch_max_iter
        self._h_tol = h_tol
        self._w_tol = w_tol
        self._shuffle = shuffle
        self._verbose = verbose
        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)
        # Floor for HALS denominators (as sklearn's EPSILON), so 0/0 cannot produce NaN.
        self._eps = torch.finfo(self._tensor_dtype).eps
        # Renormalize W rows after each pass (see _rescale_components). Only applied without
        # regularization, where the rescaling is an exact symmetry of the objective.
        self._has_reg = any(r != 0 for r in (self._l1_reg_W, self._l2_reg_W, self._l1_reg_H, self._l2_reg_H))
        self._rescale = rescale and not self._has_reg
        if rescale and self._has_reg:
            print("rescale=True ignored: regularization is on (alpha > 0), and rescaling W would change the penalty.")
        # Stopping rule. An int >= 1 checks the loss every pass and allows that many passes without a
        # new best. None or 0 keeps the original rule (loss every 10 passes, stop on the first increase).
        if patience is not None and (not isinstance(patience, int) or patience < 0):
            raise ValueError(f"patience must be None or an int >= 0, got {patience!r}.")
        self._patience = patience or None
        self._tol_window = tol_window


    def _update_one_pass(self, l1_reg_W, l2_reg_W):
        if self._shuffle:
            indices = torch.randperm(self.X.shape[0], device=self._device_type)
        else:
            indices = torch.arange(self.X.shape[0], device=self._device_type)
        A = torch.zeros((self.k, self.k), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type)

        i = 0
        num_processed = 0
        while i < indices.shape[0]:
            idx = indices[i:(i+self._minibatch_size)]
            cur_minibatch_size = idx.shape[0]
            x = self.X[idx, :]
            h = self.H[idx, :]

            # Online update H.
            WWT = self.W @ self.W.T
            xWT = x @ self.W.T

            for j in range(self._minibatch_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = xWT[:, k] - h @ WWT[:, k]
                    if self._l1_reg_H > 0.0:
                        numer -= self._l1_reg_H
                    if self._l2_reg_H > 0.0:
                        denom = (WWT[k, k] + self._l2_reg_H).clamp_min(self._eps)
                        hvec = h[:, k] * (WWT[k, k] / denom) + numer / denom
                    else:
                        hvec = h[:, k] + numer / WWT[k, k].clamp_min(self._eps)
                    # Per-entry guard: zero only the non-finite entries, not the whole column.
                    hvec = torch.nan_to_num(hvec, nan=0.0, posinf=0.0, neginf=0.0).maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(h[:, k] - hvec).max())
                    h[:, k] = hvec
                
                if j + 1 < self._minibatch_max_iter and cur_max / h.mean() < self._h_tol:
                    break

            if self._verbose:
                print(f"        Block {i // self._minibatch_size} update H iterates {j+1} iterations"
                      f"{' (hit max_iter)' if j + 1 == self._minibatch_max_iter else ''}.")
            # print(f"            max|dH|={float(cur_max):.4g}, rel={float(cur_max / h.mean()):.4g} (tol {self._h_tol}).")
            self.H[idx, :] = h

            # Update sufficient statistics A and B.
            num_after = num_processed + cur_minibatch_size

            A *= num_processed
            A += h.T @ h
            A /= num_after

            B *= num_processed
            B += h.T @ x
            B /= num_after

            num_processed = num_after

            # Online update W.
            for j in range(self._minibatch_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = B[k, :] - A[k, :] @ self.W
                    if l1_reg_W > 0.0:
                        numer -= l1_reg_W
                    if l2_reg_W > 0.0:
                        denom = (A[k, k] + l2_reg_W).clamp_min(self._eps)
                        w_new = self.W[k, :] * (A[k, k] / denom) + numer / denom
                    else:
                        w_new = self.W[k, :] + numer / A[k, k].clamp_min(self._eps)
                    # Per-entry guard: zero only the non-finite entries, not the whole row.
                    w_new = torch.nan_to_num(w_new, nan=0.0, posinf=0.0, neginf=0.0).maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(self.W[k, :] - w_new).max())
                    self.W[k, :] = w_new
                
                if j + 1 < self._minibatch_max_iter and cur_max / self.W.mean() < self._w_tol:
                    break

            if self._verbose:
                print(f"        Block {i // self._minibatch_size} update W iterates {j+1} iterations"
                      f"{' (hit max_iter)' if j + 1 == self._minibatch_max_iter else ''}.")
            # print(f"            max|dW|={float(cur_max):.4g}, rel={float(cur_max / self.W.mean()):.4g} (tol {self._w_tol}).")
            i += self._minibatch_size


    def _update_H(self):
        i = 0
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._minibatch_size), :]
            h = self.H[i:(i+self._minibatch_size), :]

            xWT = x @ self.W.T
            for j in range(self._minibatch_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = xWT[:, k] - h @ WWT[:, k]
                    if self._l1_reg_H > 0.0:
                        numer -= self._l1_reg_H
                    if self._l2_reg_H > 0.0:
                        denom = (WWT[k, k] + self._l2_reg_H).clamp_min(self._eps)
                        hvec = h[:, k] * (WWT[k, k] / denom) + numer / denom
                    else:
                        hvec = h[:, k] + numer / WWT[k, k].clamp_min(self._eps)
                    # Per-entry guard: zero only the non-finite entries, not the whole column.
                    hvec = torch.nan_to_num(hvec, nan=0.0, posinf=0.0, neginf=0.0).maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(h[:, k] - hvec).max())
                    h[:, k] = hvec

                if j + 1 < self._minibatch_max_iter and cur_max / h.mean() < self._h_tol:
                    break

            if self._verbose:
                print(f"        Final block {i // self._minibatch_size} update H iterates {j+1} iterations"
                      f"{' (hit max_iter)' if j + 1 == self._minibatch_max_iter else ''}.")
            # print(f"            max|dH|={float(cur_max):.4g}, rel={float(cur_max / h.mean()):.4g} (tol {self._h_tol}).")
            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)

            i += self._minibatch_size

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def _rescale_components(self):
        """Remove the W·H scale ambiguity: unit-L2 W rows, H columns scaled to compensate.

        W·H, and therefore the loss, is unchanged. Without this, online HALS lets a component's
        W row grow every pass while its H column shrinks, until float32 overflows (inf -> NaN).
        Called between passes, where A and B are rebuilt from scratch, so they need no rescaling.
        """
        norms = self.W.norm(dim=1)
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))  # leave all-zero rows as they are
        self.W /= norms[:, None]
        self.H *= norms[None, :]


    def _run_passes_legacy(self, l1_reg_W, l2_reg_W):
        """Original stopping rule: loss every 10 passes, stop on the first increase or a small change."""
        for i in range(self._max_epoch):
            self._update_one_pass(l1_reg_W, l2_reg_W)
            if self._rescale:
                self._rescale_components()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                print(f"Pass {i+1}, loss={self._cur_err}.")

                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} pass(es).")
                    break

                self._prev_err = self._cur_err


    def _run_passes_patience(self, l1_reg_W, l2_reg_W):
        """Loss every pass. Stop after `patience` passes without a new best loss, or when the best loss
        improves by less than tol (relative to the initial loss) over `tol_window` passes.
        W/H are restored to the best pass before returning."""
        print(f"Check loss every pass; stop after {self._patience} pass(es) without a new best, "
              f"or when the best loss improves < tol over {self._tol_window} passes.")
        best_err = self._init_err
        best_hist = collections.deque([best_err], maxlen=self._tol_window + 1)  # best loss over the last tol_window passes
        best_W, best_H = self.W.clone(), self.H.clone()
        best_pass, no_improve = 0, 0

        for i in range(self._max_epoch):
            self._update_one_pass(l1_reg_W, l2_reg_W)
            if self._rescale:
                self._rescale_components()

            self._cur_err = self._loss()
            if self._cur_err < best_err:
                best_err, best_pass, no_improve = self._cur_err, i + 1, 0
                best_W.copy_(self.W)
                best_H.copy_(self.H)
            else:
                no_improve += 1
            best_hist.append(best_err)

            if (i + 1) % 10 == 0 or no_improve == 1:
                print(f"Pass {i+1}, loss={self._cur_err}, best={best_err}, passes since best={no_improve}.")

            # Patience: the loss may rise, but not for `patience` passes in a row.
            stop_reason = None
            if no_improve >= self._patience:
                stop_reason = f"no new best loss for {self._patience} pass(es)"
            # Tolerance: same meaning as the legacy rule (change over 10 passes), measured on the best
            # loss and only while still improving, so a temporary rise cannot trigger it.
            elif no_improve == 0 and len(best_hist) == best_hist.maxlen and \
                    (best_hist[0] - best_err) / self._init_err < self._tol:
                stop_reason = f"best loss improved < tol over {self._tol_window} passes"

            if stop_reason is not None:
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es): {stop_reason}.")
                break

        if no_improve > 0:
            print(f"Restore W/H from the best pass ({best_pass}, loss={best_err}).")
            self.W.copy_(best_W)
            self.H.copy_(best_H)


    def fit(self, X):
        super().fit(X)
        assert self._beta==2, "Cannot perform online update when beta is not equal to 2!"

        # Online update.
        self._minibatch_size = min(self.X.shape[0], self._minibatch_size)

        l1_reg_W = self._l1_reg_W / self.X.shape[0]
        l2_reg_W = self._l2_reg_W / self.X.shape[0]

        print(f"Rescale W rows to unit norm after each pass: {self._rescale}")
        self.num_iters = -1
        if self._patience is None:
            self._run_passes_legacy(l1_reg_W, l2_reg_W)
        else:
            self._run_passes_patience(l1_reg_W, l2_reg_W)

        if self.num_iters < 0:
            self.num_iters = self._max_epoch
            print(f"    Not converged after {self._max_epoch} pass(es).")

        print(f"Update H")
        self._cur_err = self._update_H()
        print(f"Final loss={self._cur_err}.")