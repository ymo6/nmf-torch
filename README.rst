==============
NMF-Torch
==============

|PyPI| |Python| |License|

.. |PyPI| image:: https://img.shields.io/pypi/v/nmf-torch.svg
   :target: https://pypi.org/project/nmf-torch

.. |Python| image:: https://img.shields.io/pypi/pyversions/nmf-torch
   :target: https://pypi.org/project/nmf-torch

.. |License| image:: https://img.shields.io/github/license/lilab-bcb/nmf-torch
   :target: https://github.com/lilab-bcb/nmf-torch/blob/main/LICENSE

A PyTorch implementation on Non-negative Matrix Factorization.

.. warning::

   This is a **fork** of the original `nmf-torch <https://github.com/lilab-bcb/nmf-torch>`_ by Li Lab.
   It adds a DataLoader-based minibatch mode for memory-efficient GPU usage, a shuffle control parameter,
   and other modifications. For the original upstream package, see https://github.com/lilab-bcb/nmf-torch.


Installation
^^^^^^^^^^^^^

Install directly from this fork::

	pip install git+https://github.com/ymo6/nmf-torch.git

For the original upstream package from PyPI::

	pip install nmf-torch


Usage
^^^^^^^^

NMF
######

Given a non-negative numeric matrix ``X`` of shape M-by-N (M is number of samples, N number of features) in either numpy array or torch tensor structure, run the following code::

	from nmf_torch import run_nmf
	H, W, err = run_nmf(X, n_components=20)

will decompose ``X`` into two new non-negative matrices:

- ``H`` of shape (M, 20), representing the transformed coordinates of samples regarding the 20 components;
- ``W`` of shape (20, N), representing the composition of each component in terms of features;

along with the loss between ``X`` and its approximation ``H*W``.

Advanced Settings
++++++++++++++++++

By default, ``run_nmf`` function uses the batch HALS solver for NMF decomposition. In total, there are other solvers available in NMF-torch:

- **HALS**: Hierarchical Alternative Least Squares (`[Kimura et al., 2015]`_). The default.
- **MU**: Multiplicative Update. Set ``algo='mu'`` in ``run_nmf`` function.
- **BPP**: Alternative non-negative least squares with Block Principal Pivoting method (`[Kim & Park, 2011]`_). Set ``algo='bpp'`` in ``run_nmf`` function.

Besides, each solver has three modes:

- **batch**: Standard batch learning, loading the entire matrix into memory. The default.
- **minibatch**: Online learning using mini-batches of samples, scalable for input matrices with a large number of samples. Set ``mode='minibatch'`` in ``run_nmf`` function.
- **dataloader**: Online learning using a PyTorch DataLoader for mini-batch iteration. Unlike ``minibatch`` mode, which loads all data onto the GPU upfront, ``dataloader`` mode loads only the necessary mini-batch onto the GPU on demand and moves it back to CPU after updating the model. This avoids CUDA out-of-memory errors when the input data is too large to fit in GPU memory. Set ``mode='dataloader'`` in ``run_nmf`` function. **Note:** Only supported for the HALS solver (``algo='hals'`` or ``algo='halsvar'``); for other solvers, it falls back to ``minibatch`` mode.

The default beta loss is Frobenius (L2) distance, which is the most commonly used.
By changing ``beta_loss`` parameter in ``run_nmf`` function,
users can specify other beta loss metrics:

- **KL divergence**: ``beta_loss='kullback-leibler'`` or ``beta_loss=1.0``;
- **Itakura-Saito divergence**: ``beta_loss='itakura-saito'`` or ``beta_loss=0``;
- Or any non-negative float number.

Notice that ``minibatch`` and ``dataloader`` modes only work for L2 loss (Frobenius). If you specify other beta loss, ``run_nmf`` will automatically switch back to batch mode.

For the other parameters in ``run_nmf`` function, please type ``help(run_nmf)`` in your Python interpreter to view.

------------

Data Integration using Integrative NMF (iNMF)
################################################

In this case, we have a list of ``k`` batches, with their corresponding non-negative numeric matrices to be integrated.
Let ``X`` be such a list, and all matrices in ``X`` have the same number of features,
i.e. each X\ :sub:`i` in ``X`` has shape (|M_i|, N), where |M_i| is number of samples in batch i, and N is number of features.

The following code::

	from nmf_torch import integrative_nmf
	H, W, V, err = integrative_nmf(X, n_components=20)

will perform iNMF, which results in the following non-negative matrices:

- ``H``: List of matrices of shape (|M_i|, 20), each of which represents the transformed coordinates of samples regarding components of the corresponding batch;
- ``W`` of shape (20, N), representing the common composition (shared information) across the given batches in terms of features;
- ``V``: List of matrices of the same shape (20, N), each of which represents the batch-specific composition in terms of features of the corresponding batch,

along with the overall L2 loss between |X_i| and its approximation |H_i| \* (W + |V_i|) for each batch i.

Advanced Settings
++++++++++++++++++

Similarly as in ``run_nmf`` function above, ``integrative_nmf`` provides 2 modes (batch and minibatch) and 3 solvers: HALS, MU, and BPP.
By default, batch HALS is used. You can switch to other solvers and modes by specifying ``algo`` and ``mode`` parameters.
Note that ``dataloader`` mode is not yet supported for integrative NMF and will fall back to ``minibatch``.

There is another important parameter ``lam`` for the coefficient for regularization terms, with default value ``5.0``.
If set to ``0``, then no regularization will be applied.

Notice that only L2 loss is accepted in iNMF.

For the other parameters in ``integrative_nmf`` function, please type ``help(integrative_nmf)`` in your Python interpreter to view.

.. |M_i| replace:: M\ :sub:`i`
.. |X_i| replace:: X\ :sub:`i`
.. |H_i| replace:: H\ :sub:`i`
.. |V_i| replace:: V\ :sub:`i`
.. _[Kimura et al., 2015]: http://proceedings.mlr.press/v39/kimura14.pdf
.. _[Kim & Park, 2011]: https://www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf
