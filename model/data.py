# -*- coding: utf-8 -*-
"""
Data handling utilities - Modified for CGLUE-SOE labels
"""
# ... (之前的 imports) ...
import copy
import functools
import multiprocessing
import operator
import os
import queue
import signal
import uuid
from math import ceil
from typing import Any, List, Mapping, Optional, Tuple
import sklearn.preprocessing # Import OneHotEncoder
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from anndata import AnnData

try:
    from anndata._core.sparse_dataset import SparseDataset
except ImportError:  # Newer version of anndata
    from anndata._core.sparse_dataset import \
        BaseCompressedSparseDataset as SparseDataset

from ..num import vertex_degrees
from ..typehint import AnyArray, Array, RandomState
from ..utils import config, get_rs, logged, processes
from .nn import get_default_numpy_dtype

DATA_CONFIG = Mapping[str, Any]


# --------------------------------- Datasets -----------------------------------


@logged
class Dataset(torch.utils.data.Dataset):

    r"""
    Abstract dataset interface extending that of :class:`torch.utils.data.Dataset`

    Parameters
    ----------
    getitem_size
        Unitary fetch size for each __getitem__ call
    """

    def __init__(self, getitem_size: int = 1) -> None:
        super().__init__()
        self.getitem_size = getitem_size
        self.shuffle_seed: Optional[int] = None
        self.seed_queue: Optional[multiprocessing.Queue] = None
        self.propose_queue: Optional[multiprocessing.Queue] = None
        self.propose_cache: Mapping[int, Any] = {}

    @property
    def has_workers(self) -> bool:
        r"""
        Whether background shuffling workers have been registered
        """
        self_processes = processes[id(self)]
        pl = bool(self_processes)
        sq = self.seed_queue is not None
        pq = self.propose_queue is not None
        if not pl == sq == pq:
            raise RuntimeError("Background shuffling seems broken!")
        return pl and sq and pq

    def prepare_shuffle(self, num_workers: int = 1, random_seed: int = 0) -> None:
        r"""
        Prepare dataset for custom shuffling

        Parameters
        ----------
        num_workers
            Number of background workers for data shuffling
        random_seed
            Initial random seed (will increase by 1 with every shuffle call)
        """
        if self.has_workers:
            self.clean()
        self_processes = processes[id(self)]
        self.shuffle_seed = random_seed
        if num_workers:
            self.seed_queue = multiprocessing.Queue()
            self.propose_queue = multiprocessing.Queue()
            for i in range(num_workers):
                p = multiprocessing.Process(target=self.shuffle_worker)
                p.start()
                self.logger.debug("Started background process: %d", p.pid)
                self_processes[p.pid] = p
                self.seed_queue.put(self.shuffle_seed + i)

    def shuffle(self) -> None:
        r"""
        Custom shuffling
        """
        if self.has_workers:
            self_processes = processes[id(self)]
            self.seed_queue.put(self.shuffle_seed + len(self_processes))  # Look ahead
            while self.shuffle_seed not in self.propose_cache:
                shuffle_seed, shuffled = self.propose_queue.get()
                self.propose_cache[shuffle_seed] = shuffled
            self.accept_shuffle(self.propose_cache.pop(self.shuffle_seed))
        else:
            self.accept_shuffle(self.propose_shuffle(self.shuffle_seed))
        self.shuffle_seed += 1

    def shuffle_worker(self) -> None:
        r"""
        Background shuffle worker
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            seed = self.seed_queue.get()
            if seed is None:
                self.propose_queue.put((None, os.getpid()))
                break
            self.propose_queue.put((seed, self.propose_shuffle(seed)))

    def propose_shuffle(self, seed: int) -> Any:
        r"""
        Propose shuffling using a given random seed

        Parameters
        ----------
        seed
            Random seed

        Returns
        -------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def accept_shuffle(self, shuffled: Any) -> None:
        r"""
        Accept shuffling result

        Parameters
        ----------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def clean(self) -> None:
        r"""
        Clean up multi-process resources used in custom shuffling
        """
        self_processes = processes[id(self)]
        if not self.has_workers:
            return
        for _ in self_processes:
            self.seed_queue.put(None)
        self.propose_cache.clear()
        while self_processes:
            try:
                first, second = self.propose_queue.get(
                    timeout=config.FORCE_TERMINATE_WORKER_PATIENCE
                )
            except queue.Empty:
                break
            if first is not None:
                continue
            pid = second
            self_processes[pid].join()
            self.logger.debug("Joined background process: %d", pid)
            del self_processes[pid]
        for pid in list(
            self_processes.keys()
        ):  # If some background processes failed to exit gracefully
            self_processes[pid].terminate()
            self_processes[pid].join()
            self.logger.debug("Terminated background process: %d", pid)
            del self_processes[pid]
        self.propose_queue = None
        self.seed_queue = None

    def __del__(self) -> None:
        self.clean()


@logged
class ArrayDataset(Dataset):

    r"""
    Array dataset for :class:`numpy.ndarray` and :class:`scipy.sparse.spmatrix`
    objects. Different arrays are considered as unpaired, and thus do not need
    to have identical sizes in the first dimension. Smaller arrays are recycled.
    Also, data fetched from this dataset are automatically densified.

    Parameters
    ----------
    *arrays
        An arbitrary number of data arrays

    Note
    ----
    We keep using arrays because sparse tensors do not support slicing.
    Arrays are only converted to tensors after minibatch slicing.
    """

    def __init__(self, *arrays: Array, getitem_size: int = 1) -> None:
        super().__init__(getitem_size=getitem_size)
        self.sizes = None
        self.size = None
        self.view_idx = None
        self.shuffle_idx = None
        self.arrays = arrays

    @property
    def arrays(self) -> List[Array]:
        r"""
        Internal array objects
        """
        return self._arrays

    @arrays.setter
    def arrays(self, arrays: List[Array]) -> None:
        self.sizes = [array.shape[0] for array in arrays]
        if min(self.sizes) == 0:
            raise ValueError("Empty array is not allowed!")
        self.size = max(self.sizes)
        self.view_idx = [np.arange(s) for s in self.sizes]
        self.shuffle_idx = self.view_idx
        self._arrays = arrays

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        index = np.arange(
            index * self.getitem_size, min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(
                a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]].toarray()
            )
            if scipy.sparse.issparse(a) or isinstance(a, SparseDataset)
            else torch.as_tensor(a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]])
            for i, a in enumerate(self.arrays)
        ]

    def propose_shuffle(self, seed: int) -> List[np.ndarray]:
        rs = get_rs(seed)
        return [rs.permutation(view_idx) for view_idx in self.view_idx]

    def accept_shuffle(self, shuffled: List[np.ndarray]) -> None:
        self.shuffle_idx = shuffled

    def random_split(
        self, fractions: List[float], random_state: RandomState = None
    ) -> List["ArrayDataset"]:
        r"""
        Randomly split the dataset into multiple subdatasets according to
        given fractions.

        Parameters
        ----------
        fractions
            Fraction of each split
        random_state
            Random state

        Returns
        -------
        subdatasets
            A list of splitted subdatasets
        """
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        subdatasets = [
            ArrayDataset(*self.arrays, getitem_size=self.getitem_size)
            for _ in fractions
        ]
        for j, view_idx in enumerate(self.view_idx):
            view_idx = rs.permutation(view_idx)
            split_pos = np.round(cum_frac * view_idx.size).astype(int)
            split_idx = np.split(
                view_idx, split_pos[:-1]
            )  # Last pos produces an extra empty split
            for i, idx in enumerate(split_idx):
                subdatasets[i].sizes[j] = len(idx)
                subdatasets[i].view_idx[j] = idx
                subdatasets[i].shuffle_idx[j] = idx
        return subdatasets


@logged
class AnnDatasetWithLabels(Dataset): # <<<--- *** RENAMED CLASS HERE ***

    r"""
    Dataset for :class:`anndata.AnnData` objects with partial pairing support
    and handling for ordered labels required by CGLUE-SOE.

    Parameters
    ----------
    adatas
        An arbitrary number of configured :class:`anndata.AnnData` objects.
        Configuration must be done using `configure_dataset_cglue_soe`.
    data_configs
        Data configurations from `adata.uns[config.ANNDATA_KEY]`.
    mode
        Data mode, must be one of ``{"train", "eval"}``.
    getitem_size
        Unitary fetch size for each __getitem__ call.
    """

    def __init__(
        self,
        adatas: List[AnnData],
        data_configs: List[DATA_CONFIG],
        mode: str = "train",
        getitem_size: int = 1,
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        if mode not in ("train", "eval"):
            raise ValueError("Invalid `mode`!")
        self.mode = mode
        # Store label encoders for each dataset
        self.label_encoders: List[Optional[sklearn.preprocessing.OneHotEncoder]] = []
        # Call setters which will trigger data extraction including labels
        self.adatas = adatas
        self.data_configs = data_configs # This setter now handles labels

    @property
    def adatas(self) -> List[AnnData]:
        return self._adatas

    @property
    def data_configs(self) -> List[DATA_CONFIG]:
        return self._data_configs

    @adatas.setter
    def adatas(self, adatas: List[AnnData]) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        if min(self.sizes) == 0:
            raise ValueError("Empty dataset is not allowed!")
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs: List[DATA_CONFIG]) -> None:
        if len(data_configs) != len(self.adatas):
            raise ValueError(
                "Number of data configs must match " "the number of datasets!"
            )

        # Initialize label encoders based on config
        self.label_encoders = []
        for cfg in data_configs:
            if cfg.get("use_label"):
                # Ensure categories are stored and ordered correctly in the config
                ordered_categories = cfg.get("labels_ordered")
                if ordered_categories is None:
                     raise ValueError("Data config must contain 'labels_ordered' for CGLUE-SOE.")
                encoder = sklearn.preprocessing.OneHotEncoder(
                    categories=[ordered_categories], # Pass ordered categories
                    sparse_output=False,
                    handle_unknown='ignore' # Or 'error' if preferred
                )
                # Fit the encoder with dummy data to initialize it correctly
                # This assumes categories are properly defined in the config
                dummy_labels = np.array(ordered_categories).reshape(-1, 1)
                encoder.fit(dummy_labels)
                self.label_encoders.append(encoder)
            else:
                self.label_encoders.append(None) # No label encoder for this dataset

        # Extract data including labels now
        self.data_idx, self.extracted_data = self._extract_data(data_configs)
        self.view_idx = (
            pd.concat([data_idx.to_series() for data_idx in self.data_idx])
            .drop_duplicates()
            .to_numpy()
        )
        self.size = self.view_idx.size
        self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx)
        self._data_configs = data_configs

    # --- Methods related to shuffling (_get_idx_pmsk, propose_shuffle, accept_shuffle) ---
    # These likely DO NOT need changes, as they operate on the unified view_idx
    # and generate indices (shuffle_idx) and masks (shuffle_pmsk) that are
    # applied commonly during __getitem__. The labels will be indexed using
    # the same shuffle_idx as the data.

    def _get_idx_pmsk( # No changes needed here usually
        self,
        view_idx: np.ndarray,
        random_fill: bool = False,
        random_state: RandomState = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rs = get_rs(random_state) if random_fill else None
        shuffle_idx, shuffle_pmsk = [], []
        for data_idx in self.data_idx:
            idx = data_idx.get_indexer(view_idx)
            pmsk = idx >= 0
            n_true = pmsk.sum()
            if n_true == 0: # Handle case where a dataset has no overlap with view_idx
                 # Fill with a default index (e.g., 0) or handle appropriately
                 idx[:] = 0 # Or raise error, depending on desired behavior
                 pmsk[:] = False
                 self.logger.warning("Dataset has no overlap with current view_idx. Filling indices with 0.")
            else:
                 n_false = pmsk.size - n_true
                 if n_false > 0:
                     fill_indices = (
                         rs.choice(idx[pmsk], n_false, replace=True)
                         if random_fill
                         else idx[pmsk][np.mod(np.arange(n_false), n_true)]
                     )
                     idx[~pmsk] = fill_indices
            shuffle_idx.append(idx)
            shuffle_pmsk.append(pmsk)
        return np.stack(shuffle_idx, axis=1), np.stack(shuffle_pmsk, axis=1)

    def propose_shuffle(self, seed: int) -> Tuple[np.ndarray, np.ndarray]: # No changes needed
        rs = get_rs(seed)
        view_idx = rs.permutation(self.view_idx)
        return self._get_idx_pmsk(view_idx, random_fill=True, random_state=rs)

    def accept_shuffle(self, shuffled: Tuple[np.ndarray, np.ndarray]) -> None: # No changes needed
        self.shuffle_idx, self.shuffle_pmsk = shuffled

    # --- __len__ remains the same ---
    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    # --- Modify __getitem__ to return labels ---
    def __getitem__(self, index: int) -> List[torch.Tensor]:
        """ Returns: [x_mod1, x_mod2, ..., xrep_mod1, ..., xbch_mod1, ..., y_onehot_mod1, ..., xdwt_mod1, ..., pmsk] """
        s = slice(
            index * self.getitem_size, min((index + 1) * self.getitem_size, self.size)
        )
        shuffle_idx = self.shuffle_idx[s].T # Shape: [num_datasets, batch_slice_size]
        shuffle_pmsk = self.shuffle_pmsk[s] # Shape: [batch_slice_size, num_datasets]

        # Unpack extracted data (now includes y_onehot)
        x_list, xrep_list, xbch_list, y_onehot_list, xdwt_list = self.extracted_data

        items = []
        # Add x data
        items.extend([
            torch.as_tensor(self._index_array(data, idx))
            for data, idx in zip(x_list, shuffle_idx)
        ])
        # Add xrep data
        items.extend([
            torch.as_tensor(self._index_array(data, idx))
            for data, idx in zip(xrep_list, shuffle_idx)
        ])
        # Add xbch data
        items.extend([
             # Batch indices are usually simple arrays, direct indexing is fine
            torch.as_tensor(data[idx])
            for data, idx in zip(xbch_list, shuffle_idx)
        ])
        # Add y_onehot data (NEW)
        items.extend([
            torch.as_tensor(self._index_array(data, idx))
            for data, idx in zip(y_onehot_list, shuffle_idx)
        ])
        # Add xdwt data
        items.extend([
            # Weights are usually simple arrays
            torch.as_tensor(data[idx])
            for data, idx in zip(xdwt_list, shuffle_idx)
        ])
        # Add pairing mask
        items.append(torch.as_tensor(shuffle_pmsk))

        return items

    # --- _index_array remains the same ---
    @staticmethod
    def _index_array(arr: AnyArray, idx: np.ndarray) -> np.ndarray:
        # ... (keep existing implementation)
        if isinstance(arr, (h5py.Dataset, SparseDataset)):
            # Ensure indices are unique and sorted for efficient HDF5/SparseDataset access
            unique_idx, inverse_map = np.unique(idx, return_inverse=True)
            # Access unique indices
            arr_subset = arr[unique_idx.tolist()]
            # Map back to original batch order
            arr = arr_subset[inverse_map]
        else:
            arr = arr[idx]
        return arr.toarray() if scipy.sparse.issparse(arr) else arr


    # --- Modify _extract_data methods ---
    def _extract_data(
        self, data_configs: List[DATA_CONFIG]
    ) -> Tuple[
        List[pd.Index],
        Tuple[ # Return tuple now includes y_onehot_list
            List[AnyArray], # x
            List[AnyArray], # xrep
            List[AnyArray], # xbch
            List[AnyArray], # y_onehot (NEW)
            List[AnyArray]  # xdwt
        ],
    ]:
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)

    def _extract_data_train(
        self, data_configs: List[DATA_CONFIG]
    ) -> Tuple[
        List[pd.Index],
        Tuple[List[AnyArray], List[AnyArray], List[AnyArray], List[AnyArray], List[AnyArray]], # Added y_onehot
    ]:
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xrep = [
            self._extract_xrep(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = [
            self._extract_xbch(adata, data_config)
            for i, (adata, data_config) in enumerate(zip(self.adatas, data_configs))
        ]
        # Extract labels and one-hot encode them
        y_onehot = [
            self._extract_y_onehot(adata, data_config, i) # Use helper
            for i, (adata, data_config) in enumerate(zip(self.adatas, data_configs))
        ]
        xdwt = [
            self._extract_xdwt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        # Return tuple including y_onehot
        return xuid, (x, xrep, xbch, y_onehot, xdwt)

    def _extract_data_eval(
        self, data_configs: List[DATA_CONFIG]
    ) -> Tuple[
        List[pd.Index],
        Tuple[List[AnyArray], List[AnyArray], List[AnyArray], List[AnyArray], List[AnyArray]], # Added y_onehot
    ]:
        default_dtype = get_default_numpy_dtype()
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xrep = [
            self._extract_xrep(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        # For eval, we primarily need x or xrep, and labels for the conditional encoder
        x = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            if xrep_.size else self._extract_x(adata, data_config)
            for adata, data_config, xrep_ in zip(self.adatas, data_configs, xrep)
        ]
        y_onehot = [ # Still need labels for conditional encoder during eval
            self._extract_y_onehot(adata, data_config, i)
            for i, (adata, data_config) in enumerate(zip(self.adatas, data_configs))
        ]
        # Other fields can be empty for eval mode if not used by the encoder/model
        xbch = [np.empty((adata.shape[0], 0), dtype=int) for adata in self.adatas]
        xdwt = [np.empty((adata.shape[0], 0), dtype=default_dtype) for adata in self.adatas]
        return xuid, (x, xrep, xbch, y_onehot, xdwt)

    # --- Helper methods for extracting specific data fields ---
    # Keep existing _extract_x, _extract_xrep, _extract_xbch, _extract_xdwt, _extract_xuid

    def _extract_x(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        # ... (keep existing implementation)
        default_dtype = get_default_numpy_dtype()
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        # Ensure adata has features in the correct order
        current_features = adata.var_names.to_list()
        if current_features != features:
             missing_features = set(features) - set(current_features)
             if missing_features:
                 raise ValueError(f"Dataset missing required features: {missing_features}")
             adata = adata[:, features] # Reorder and subset

        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(f"Configured data layer '{use_layer}' not found!")
            x = adata.layers[use_layer]
        else:
            x = adata.X
        if hasattr(x, 'dtype') and x.dtype.type is not default_dtype:
            if isinstance(x, (h5py.Dataset, SparseDataset)):
                # For backed data, conversion might load everything to memory.
                # Consider checking dtype earlier or handling differently.
                self.logger.warning(f"Backed data detected with dtype {x.dtype}. Attempting conversion to {default_dtype}, may load data.")
                # This conversion might be problematic for large backed data.
                try:
                    x = x[:].astype(default_dtype) # Attempt conversion
                except Exception as e:
                    raise RuntimeError(f"Failed to convert backed data to {default_dtype}: {e}")
            else:
                x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr() # Convert to CSR for potentially faster row slicing
        return x


    def _extract_xrep(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        # ... (keep existing implementation)
        default_dtype = get_default_numpy_dtype()
        use_rep = data_config["use_rep"]
        rep_dim = data_config["rep_dim"]
        if use_rep:
            if use_rep not in adata.obsm:
                raise ValueError(f"Configured data representation '{use_rep}' not found!")
            xrep = np.asarray(adata.obsm[use_rep]).astype(default_dtype)
            if rep_dim is not None and xrep.shape[1] != rep_dim:
                 # Check rep_dim consistency if it was set during configuration
                 raise ValueError(f"Input representation dim {xrep.shape[1]} != configured {rep_dim}!")
            return xrep
        # Return empty array with correct shape if no rep is used
        return np.empty((adata.shape[0], 0), dtype=default_dtype)


    def _extract_xbch(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        # ... (keep existing implementation)
        use_batch = data_config.get("use_batch") # Use .get for safety
        batches = data_config.get("batches")
        if use_batch and batches is not None:
            if use_batch not in adata.obs:
                raise ValueError(f"Configured data batch '{use_batch}' not found!")
            # Use pandas Categorical for robust index lookup
            batch_cat = pd.Categorical(adata.obs[use_batch], categories=batches)
            if batch_cat.isna().any():
                 raise ValueError(f"Missing batch values found in column '{use_batch}'.")
            return batch_cat.codes # Return numerical codes
        # Return array of zeros if no batch info
        return np.zeros(adata.shape[0], dtype=int)


    def _extract_y_onehot(self, adata: AnnData, data_config: DATA_CONFIG, encoder_idx: int) -> AnyArray:
        """Extracts and one-hot encodes labels."""
        use_label = data_config.get("use_label")
        if use_label:
            if use_label not in adata.obs:
                raise ValueError(f"Configured label column '{use_label}' not found!")
            label_encoder = self.label_encoders[encoder_idx]
            if label_encoder is None:
                 raise RuntimeError(f"Label encoder not initialized for dataset index {encoder_idx}.")
            # Ensure the column is categorical with the correct order before transform
            labels_cat = adata.obs[use_label].astype(pd.CategoricalDtype(categories=label_encoder.categories_[0], ordered=True))
            if labels_cat.isna().any():
                 raise ValueError(f"Missing label values found in column '{use_label}'.")
            # Transform using the pre-fitted encoder
            y_onehot = label_encoder.transform(labels_cat.to_numpy().reshape(-1, 1))
            return y_onehot.astype(get_default_numpy_dtype()) # Ensure correct dtype
        else:
            # Return empty array if no labels are configured (though CGLUE-SOE needs them)
            self.logger.warning("No label column configured via 'use_label'. CGLUE-SOE requires labels.")
            return np.empty((adata.shape[0], 0), dtype=get_default_numpy_dtype())

    def _extract_xdwt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        # ... (keep existing implementation)
        default_dtype = get_default_numpy_dtype()
        use_dsc_weight = data_config.get("use_dsc_weight")
        if use_dsc_weight:
            if use_dsc_weight not in adata.obs:
                raise ValueError(f"Configured discriminator weight '{use_dsc_weight}' not found!")
            xdwt = adata.obs[use_dsc_weight].to_numpy().astype(default_dtype)
            # Normalize weights per dataset/batch
            xdwt_sum = xdwt.sum()
            if xdwt_sum > 0:
                 xdwt = xdwt / xdwt_sum * xdwt.size # Normalize to mean 1
            else:
                 self.logger.warning(f"Sum of discriminator weights in '{use_dsc_weight}' is zero. Using equal weights.")
                 xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        else:
            xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        return xdwt


    def _extract_xuid(self, adata: AnnData, data_config: DATA_CONFIG) -> pd.Index:
        # ... (keep existing implementation)
        if data_config.get("use_obs_names", False): # Default to False if not present
            xuid = adata.obs_names.to_numpy()
        else:
            self.logger.debug("Generating random xuid...")
            xuid = np.array([uuid.uuid4().hex for _ in range(adata.shape[0])])
        if len(set(xuid)) != xuid.size:
            # Consider adding more context to the error message
            counts = pd.Series(xuid).value_counts()
            duplicates = counts[counts > 1].index.tolist()
            raise ValueError(f"Non-unique cell ID found! Duplicates: {duplicates[:5]}...") # Show first 5 duplicates
        return pd.Index(xuid)


    # --- random_split should work as is, as it operates on view_idx ---
    def random_split(
        self, fractions: List[float], random_state: RandomState = None
    ) -> List["AnnDatasetWithLabels"]: # Return type updated
        # ... (keep existing implementation)
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) > 1.0 + 1e-6: # Allow for small floating point errors
            raise ValueError("Fractions sum to more than 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        view_idx = rs.permutation(self.view_idx)
        # Ensure split positions cover the whole range correctly
        split_pos = np.round(cum_frac * view_idx.size).astype(int)
        split_pos = np.unique(np.concatenate(([0], split_pos))) # Ensure 0 is start, unique sorted positions
        if split_pos[-1] < view_idx.size: # Ensure last split goes to the end
             split_pos = np.append(split_pos, view_idx.size)

        split_idx = [view_idx[split_pos[i]:split_pos[i+1]] for i in range(len(split_pos)-1)]

        subdatasets = []
        for idx in split_idx:
            if len(idx) == 0: continue # Skip empty splits
            # Create a shallow copy first
            sub = copy.copy(self)
            # Deep copy mutable attributes like lists/dicts if necessary, but label_encoders might be ok shallow
            # Re-initialize specific attributes based on the split index 'idx'
            sub.view_idx = idx
            sub.size = idx.size
            # Recompute shuffle indices and mask for the subset
            sub.shuffle_idx, sub.shuffle_pmsk = sub._get_idx_pmsk(idx)
            subdatasets.append(sub)
        return subdatasets


@logged
class GraphDataset(Dataset):

    r"""
    Dataset for graphs with support for negative sampling

    Parameters
    ----------
    graph
        Graph object
    vertices
        Indexer of graph vertices
    neg_samples
        Number of negative samples per edge
    weighted_sampling
        Whether to do negative sampling based on vertex importance
    deemphasize_loops
        Whether to deemphasize self-loops when computing vertex importance
    getitem_size
        Unitary fetch size for each __getitem__ call

    Note
    ----
    Custom shuffling performs negative sampling.
    """

    def __init__(
        self,
        graph: nx.Graph,
        vertices: pd.Index,
        neg_samples: int = 1,
        weighted_sampling: bool = True,
        deemphasize_loops: bool = True,
        getitem_size: int = 1,
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        self.eidx, self.ewt, self.esgn = self.graph2triplet(graph, vertices)
        self.eset = {(i, j, s) for (i, j), s in zip(self.eidx.T, self.esgn)}

        self.vnum = self.eidx.max() + 1
        if weighted_sampling:
            if deemphasize_loops:
                non_loop = self.eidx[0] != self.eidx[1]
                eidx = self.eidx[:, non_loop]
                ewt = self.ewt[non_loop]
            else:
                eidx = self.eidx
                ewt = self.ewt
            degree = vertex_degrees(eidx, ewt, vnum=self.vnum, direction="both")
        else:
            degree = np.ones(self.vnum, dtype=self.ewt.dtype)
        degree_sum = degree.sum()
        if degree_sum:
            self.vprob = degree / degree_sum  # Vertex sampling probability
        else:  # Possible when `deemphasize_loops` is set on a loop-only graph
            self.vprob = np.ones(self.vnum, dtype=self.ewt.dtype) / self.vnum

        effective_enum = self.ewt.sum()
        # Handle case where effective_enum might be zero
        if effective_enum > 0:
            self.eprob = self.ewt / effective_enum  # Edge sampling probability
        else:
            self.eprob = np.zeros_like(self.ewt) # Avoid division by zero
        self.effective_enum = round(effective_enum)


        self.neg_samples = neg_samples
        self.size = self.effective_enum * (1 + self.neg_samples)
        self.samp_eidx: Optional[np.ndarray] = None
        self.samp_ewt: Optional[np.ndarray] = None
        self.samp_esgn: Optional[np.ndarray] = None

    def graph2triplet(
        self,
        graph: nx.Graph,
        vertices: pd.Index,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Convert graph object to graph triplet

        Parameters
        ----------
        graph
            Graph object
        vertices
            Graph vertices

        Returns
        -------
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)
        """
        graph = nx.MultiDiGraph(
            graph
        )  # Convert undirected to bi-directed, while keeping multi-edges

        default_dtype = get_default_numpy_dtype()
        i, j, w, s = [], [], [], []
        for k, v in dict(graph.edges).items():
            i.append(k[0])
            j.append(k[1])
            w.append(v["weight"])
            s.append(v["sign"])
        eidx = np.stack([vertices.get_indexer(i), vertices.get_indexer(j)]).astype(
            np.int64
        )
        if eidx.min() < 0:
            missing_vertices = set(i).union(set(j)) - set(vertices)
            raise ValueError(f"Graph contains vertices not in the provided vertices index: {list(missing_vertices)[:5]}...")
        ewt = np.asarray(w).astype(default_dtype)
        if (ewt <= 0).any() or (ewt > 1).any():
             # Allow weight of 1 for self-loops? Original GLUE might. Check usage.
             # Let's relax the upper bound check slightly for potential floating point issues.
             if (ewt <= 0).any() or (ewt > 1.0001).any():
                 raise ValueError("Invalid edge weight! Must be in (0, 1].")
        esgn = np.asarray(s).astype(default_dtype)
        if set(esgn).difference({-1.0, 1.0}): # Check float values
            raise ValueError(f"Invalid edge sign! Must be -1 or 1. Found: {set(esgn)}")
        return eidx, ewt, esgn

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        if self.samp_eidx is None:
             # Handle case where shuffle hasn't been called yet
             self.shuffle() # Perform initial shuffle if needed
        s = slice(
            index * self.getitem_size, min((index + 1) * self.getitem_size, self.size)
        )
        # Ensure tensors are returned even if slice is empty
        eidx_slice = self.samp_eidx[:, s] if self.samp_eidx.shape[1] > 0 else torch.empty((2, 0), dtype=torch.int64)
        ewt_slice = self.samp_ewt[s] if self.samp_ewt.size > 0 else torch.empty((0,), dtype=torch.get_default_dtype())
        esgn_slice = self.samp_esgn[s] if self.samp_esgn.size > 0 else torch.empty((0,), dtype=torch.get_default_dtype())

        return [
            torch.as_tensor(eidx_slice),
            torch.as_tensor(ewt_slice),
            torch.as_tensor(esgn_slice),
        ]


    def propose_shuffle(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.effective_enum == 0: # Handle empty graph case
             return np.empty((2,0), dtype=np.int64), np.empty(0, dtype=self.ewt.dtype), np.empty(0, dtype=self.esgn.dtype)

        (pi, pj), pw, ps = self.eidx, self.ewt, self.esgn
        rs = get_rs(seed)
        # Ensure probabilities sum to 1, handle potential floating point issues
        eprob_norm = self.eprob / self.eprob.sum() if self.eprob.sum() > 0 else np.ones_like(self.eprob) / len(self.eprob)

        psamp = rs.choice(
            self.ewt.size, self.effective_enum, replace=True, p=eprob_norm
        )
        pi_, pj_, pw_, ps_ = pi[psamp], pj[psamp], pw[psamp], ps[psamp]
        pw_ = np.ones_like(pw_) # Sampled edges represent positive examples (weight 1 in loss)
        ni_ = np.tile(pi_, self.neg_samples)
        nw_ = np.zeros(pw_.size * self.neg_samples, dtype=pw_.dtype) # Negative examples have weight 0
        ns_ = np.tile(ps_, self.neg_samples) # Sign doesn't matter for negative? Or use opposite? Assume 1 for simplicity.

        # Sample negative targets based on vertex probability
        vprob_norm = self.vprob / self.vprob.sum() if self.vprob.sum() > 0 else np.ones_like(self.vprob) / len(self.vprob)
        nj_ = rs.choice(
            self.vnum, pj_.size * self.neg_samples, replace=True, p=vprob_norm
        )

        # Resample if negative sample is actually a positive sample
        remain_mask = np.array([item in self.eset for item in zip(ni_, nj_, ns_)])
        remain_indices = np.where(remain_mask)[0]
        attempts = 0
        max_attempts = 10 # Prevent potential infinite loop
        while remain_indices.size > 0 and attempts < max_attempts:
            self.logger.debug(f"Resampling {remain_indices.size} negative edges (attempt {attempts+1})...")
            newnj = rs.choice(self.vnum, remain_indices.size, replace=True, p=vprob_norm)
            nj_[remain_indices] = newnj
            # Update the mask only for the indices that were resampled
            remain_mask_updated = np.array([item in self.eset for item in zip(ni_[remain_indices], newnj, ns_[remain_indices])])
            # Update the global mask and the indices to check next time
            remain_mask[remain_indices] = remain_mask_updated
            remain_indices = np.where(remain_mask)[0]
            attempts += 1
        if attempts == max_attempts and remain_indices.size > 0:
             self.logger.warning(f"Could not find valid negative samples for {remain_indices.size} edges after {max_attempts} attempts. Graph might be too dense.")
             # Decide how to handle this: remove these edges, keep them as is?
             # For now, let's keep them, but they might introduce noise.
             pass


        idx = np.stack([np.concatenate([pi_, ni_]), np.concatenate([pj_, nj_])])
        w = np.concatenate([pw_, nw_])
        s = np.concatenate([ps_, ns_])
        perm = rs.permutation(idx.shape[1])
        return idx[:, perm], w[perm], s[perm]

    def accept_shuffle(
        self, shuffled: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        self.samp_eidx, self.samp_ewt, self.samp_esgn = shuffled


# ------------------------------- Data loaders ---------------------------------


class DataLoader(torch.utils.data.DataLoader):

    r"""
    Custom data loader that manually shuffles the internal dataset before each
    round of iteration (see :class:`torch.utils.data.DataLoader` for usage)
    """

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        # Ensure collate_fn is passed if provided, otherwise set default
        collate_fn = kwargs.pop('collate_fn', None)
        super().__init__(dataset, **kwargs)
        # Set collate_fn after super init to avoid conflicts
        if collate_fn is None:
             self.collate_fn = (
                 self._collate_graph if isinstance(dataset, GraphDataset) else self._collate
             )
        else:
             self.collate_fn = collate_fn # Use user-provided collate_fn

        # Shuffle attribute might not exist if shuffle=False in kwargs
        self.shuffle = kwargs.get("shuffle", False)


    def __iter__(self) -> "DataLoader":
        if self.shuffle and isinstance(self.dataset, Dataset): # Check if dataset supports shuffle
            self.dataset.shuffle()  # Customized shuffling
        # Create a new iterator from the DataLoader mechanism
        return super().__iter__()

    @staticmethod
    def _collate(batch):
        """Default collate for non-graph datasets (like AnnDatasetWithLabels)."""
        # Assuming batch is a list of lists/tuples from __getitem__
        # We need to zip and then concatenate each corresponding element
        num_elements = len(batch[0]) # Number of tensors returned by __getitem__
        collated = []
        for i in range(num_elements):
            elements_to_cat = [item[i] for item in batch]
            # Handle potential empty tensors before concatenation
            non_empty_elements = [e for e in elements_to_cat if e.numel() > 0]
            if not non_empty_elements:
                 # If all tensors for this position are empty, append an empty tensor
                 # Need to determine the correct shape and dtype if possible
                 # For simplicity, let's assume the first item gives the structure
                 ref_tensor = elements_to_cat[0]
                 collated.append(torch.empty((0, *ref_tensor.shape[1:]), dtype=ref_tensor.dtype, device=ref_tensor.device))
            else:
                 collated.append(torch.cat(non_empty_elements, dim=0))
        return tuple(collated)


    @staticmethod
    def _collate_graph(batch):
        """Collate function specifically for GraphDataset."""
        # Batch is a list of tuples: [(eidx1, ewt1, esgn1), (eidx2, ewt2, esgn2), ...]
        eidx, ewt, esgn = zip(*batch)
        # Filter out empty tensors before concatenating
        eidx_non_empty = [e for e in eidx if e.numel() > 0]
        ewt_non_empty = [w for w in ewt if w.numel() > 0]
        esgn_non_empty = [s for s in esgn if s.numel() > 0]

        collated_eidx = torch.cat(eidx_non_empty, dim=1) if eidx_non_empty else torch.empty((2, 0), dtype=torch.int64)
        collated_ewt = torch.cat(ewt_non_empty, dim=0) if ewt_non_empty else torch.empty((0,), dtype=torch.get_default_dtype())
        collated_esgn = torch.cat(esgn_non_empty, dim=0) if esgn_non_empty else torch.empty((0,), dtype=torch.get_default_dtype())

        return collated_eidx, collated_ewt, collated_esgn


class ParallelDataLoader:

    r"""
    Parallel data loader

    Parameters
    ----------
    *data_loaders
        An arbitrary number of data loaders
    cycle_flags
        Whether each data loader should be cycled in case they are of
        different lengths, by default none of them are cycled.
    """

    def __init__(
        self, *data_loaders: DataLoader, cycle_flags: Optional[List[bool]] = None
    ) -> None:
        cycle_flags = cycle_flags or [False] * len(data_loaders)
        if len(cycle_flags) != len(data_loaders):
            raise ValueError("Invalid cycle flags!")
        self.cycle_flags = cycle_flags
        self.data_loaders = list(data_loaders)
        self.num_loaders = len(self.data_loaders)
        self.iterators = None

    def __iter__(self) -> "ParallelDataLoader":
        self.iterators = [iter(loader) for loader in self.data_loaders]
        return self

    def _next(self, i: int) -> List[torch.Tensor]:
        try:
            return next(self.iterators[i])
        except StopIteration as e:
            if self.cycle_flags[i]:
                # self.logger.debug(f"Cycling DataLoader {i}...") # <<<--- *** REMOVED LOGGER CALL ***
                self.iterators[i] = iter(self.data_loaders[i])
                # Need to handle case where the cycled loader is empty
                try:
                    return next(self.iterators[i])
                except StopIteration:
                     # self.logger.warning(f"DataLoader {i} is empty even after cycling.") # <<<--- *** REMOVED LOGGER CALL ***
                     # If the loader is empty even after cycling, raise the original error
                     raise e
            raise e

    def __next__(self) -> List[torch.Tensor]:
        # Use functools.reduce and operator.add for cleaner concatenation
        # The result should be a single list/tuple of tensors, not nested lists
        next_batches = [self._next(i) for i in range(self.num_loaders)]
        # Assuming each self._next(i) returns a list/tuple of tensors
        # We want to concatenate corresponding elements across loaders
        # Example: [[x1, y1], [x2, y2]] -> [cat(x1,x2), cat(y1,y2)] - This is wrong, ParallelDataLoader should yield combined list
        # Correct: [[x1, y1], [g1, g2, g3]] -> [x1, y1, g1, g2, g3]
        return functools.reduce(operator.add, next_batches)

# Make the modified dataset available
__all__ = ["Dataset", "ArrayDataset", "AnnDatasetWithLabels", "GraphDataset", "DataLoader", "ParallelDataLoader"] # Export modified class
