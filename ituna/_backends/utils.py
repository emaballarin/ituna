from abc import ABC
from abc import abstractmethod
import contextlib
from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

from filelock import FileLock
import joblib
import numpy as np
import sklearn.base
import typeguard

try:
    from config_dataclass import Configurable
except ImportError:
    Configurable = None
    warnings.warn("config_dataclass is not available, saving/loading Configurable objects will not be available")

# Try to import torch and set up torch support
try:
    import torch
except ImportError:
    torch = None
    warnings.warn("torch is not available, saving/loading torch tensors will not be available")

from ituna import config


def iter_recursive(d):
    """
    Recursively iterates over dictionary or list
    yielding the values if they are not a dict or list
    """
    if isinstance(d, dict):
        for key, value in d.items():
            yield from iter_recursive(value)
    elif isinstance(d, list):
        for value in d:
            yield from iter_recursive(value)
    else:
        yield d


def hash_recursive(d):
    """
    Recursively iterates over a dictionary or list
    hasing the values that are not a dict or list
    storing the values in a flat list
    """
    value_dict = {}
    if isinstance(d, dict):
        hashed_dict = {}
        for key, value in d.items():
            hashed_dict[key], _value_dict = hash_recursive(value)
            value_dict.update(_value_dict)
        return hashed_dict, value_dict
    elif isinstance(d, list):
        hashed_list = []
        for value in d:
            hashed_value, _value_dict = hash_recursive(value)
            hashed_list.append(hashed_value)
            value_dict.update(_value_dict)
        return hashed_list, value_dict
    else:
        value_hash = hash_object(d)
        return value_hash, {value_hash: d}


def inverse_hash_recursive(hash_dict, value_dict):
    """
    Inverse of hash_recursive
    """
    if isinstance(hash_dict, dict):
        return {key: inverse_hash_recursive(item, value_dict) for key, item in hash_dict.items()}
    elif isinstance(hash_dict, list):
        return [inverse_hash_recursive(item, value_dict) for item in hash_dict]
    elif isinstance(hash_dict, str):
        # found a hash, return the value
        return value_dict[hash_dict]
    else:
        raise ValueError(f"Unsupported hash type: {type(hash_dict).__name__}")


def is_model_reference(data_object: Any) -> bool:
    return hasattr(data_object, "_datajoint_key")


class DataArguments:
    """
    This class acts as a wrapper around the data arguments which may be passed
    to a .fit(), .transform() or .score() method
    """

    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.hash_dict, self.data_objects = hash_recursive(
            dict(
                args=self.args,
                kwargs=self.kwargs,
            )
        )
        self.hash = hash_dict(self.hash_dict)

    @classmethod
    def build_path(cls, root_dir: Union[str, Path], value_hash: str):
        return Path(root_dir) / value_hash

    def save_data_objects(self, root_dir: Union[str, Path], overwrite: bool = False) -> Dict[str, Path]:
        root_dir = Path(root_dir)
        save_paths = {}
        for value_hash, value in self.data_objects.items():
            if value is None:
                continue
            if is_model_reference(value):
                save_paths[value_hash] = None
                continue
            save_paths[value_hash] = self.build_path(root_dir, value_hash)
            store_data(
                data_path=save_paths[value_hash],
                data=value,
                overwrite=overwrite,
            )
        return save_paths

    def save(self, path: Union[str, Path]):
        path = Path(path)
        root_dir = path.parent
        self.save_data_objects(root_dir)
        save_json(path, self.hash_dict)

    @classmethod
    def load_data_objects(
        cls,
        root_dir: Union[str, Path],
        hash_dict: dict,
        load_fn: Optional[Callable] = None,
    ) -> dict:
        """

        Args:
            root_dir: The root directory to load the data objects from.
            hash_dict: The hash dictionary of the data objects.
            load_fn: The function to load the data objects.
                Assumes the function takes a single argument: the hash of the data object
        """

        def default_load_fn(value_hash):
            return load_data(cls.build_path(root_dir, value_hash))

        if load_fn is None:
            load_fn = default_load_fn

        root_dir = Path(root_dir)
        data_objects = {}
        for value_hash in iter_recursive(hash_dict):
            if value_hash == "None":
                data_objects[value_hash] = None
            else:
                data_objects[value_hash] = load_fn(value_hash)
        return inverse_hash_recursive(hash_dict, data_objects)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
    ):
        path = Path(path)
        root_dir = path.parent
        hash_dict = load_json(path)
        reconstructed_dict = cls.load_data_objects(root_dir, hash_dict)
        return cls(
            *reconstructed_dict["args"],
            **reconstructed_dict["kwargs"],
        )

    @classmethod
    def from_hash_dict(cls, hash_dict: dict, root_dir: Union[str, Path], **kwargs) -> "DataArguments":
        root_dir = Path(root_dir)
        data_objects = cls.load_data_objects(root_dir, hash_dict, **kwargs)
        return cls(
            *data_objects["args"],
            **data_objects["kwargs"],
        )


@typeguard.typechecked
def hash_str(s: str) -> str:
    """Generate a short hash from a string."""
    hash_value = hashlib.md5(s.encode()).hexdigest()
    return hash_value


@typeguard.typechecked
def hash_numpy(arr: np.ndarray) -> str:
    """
    Generate a hash from a numpy array using byte view.
    See https://stackoverflow.com/questions/806151/how-to-hash-a-large-object-dataset-in-python/806342#806342
    """
    contiguous_arr = np.ascontiguousarray(arr)
    byte_view = contiguous_arr.view(np.uint8)
    hash_value = hashlib.md5(byte_view).hexdigest()
    return hash_value


@typeguard.typechecked
def hash_sklearn(
    model: sklearn.base.BaseEstimator,
    **kwargs,
) -> str:
    """Generate a hash from a sklearn model based on it's class name and
    parameters.
    Adds any additional kwargs to the dict that is hashed.
    """
    cls = model.__class__
    class_name = cls.__name__
    module_name = cls.__module__
    long_name = f"{module_name}.{class_name}"
    dict_to_hash = model.get_params()
    dict_to_hash["__class__"] = long_name

    for key, value in kwargs.items():
        assert key not in dict_to_hash, f"Key {key} is already in the model parameters"
        dict_to_hash[key] = value
    return hash_dict(dict_to_hash)


def hash_model(model: Any) -> str:
    if isinstance(model, sklearn.base.BaseEstimator):
        return hash_sklearn(model)
    else:
        raise TypeError(f"No hashing method for model type {type(model).__name__}")


def hash_data(data) -> str:
    if isinstance(data, DataArguments):
        return data.hash
    elif isinstance(data, dict):
        raise TypeError("Hashing of data dicts only supported via DataArguments wrapper")
    elif isinstance(data, list):
        raise TypeError("Hashing of data lists only supported via DataArguments wrapper")
    else:
        return hash_object(data)


@typeguard.typechecked
def hash_dict(d: dict) -> str:
    """Generate a hash from a dictionary."""

    dict_to_hash = d.copy()
    # check each key for data types that are json serializable
    # if not json serializable, pass to hash_object
    for key, value in dict_to_hash.items():
        if not isinstance(value, (int, float, str, bool, list, tuple)):
            dict_to_hash[key] = hash_object(value)
    # sort
    sorted_items = sorted(dict_to_hash.items())
    # json dump
    json_str = json.dumps(sorted_items, sort_keys=True)
    return hash_str(json_str)


@typeguard.typechecked
def hash_object(o: object) -> str:
    """Generate a short hash from an object."""
    # Check if object is a numpy array
    if isinstance(o, np.ndarray):
        return hash_numpy(o)
    elif isinstance(o, sklearn.base.BaseEstimator):
        return hash_sklearn(o)
    elif isinstance(o, dict):
        return hash_dict(o)
    elif Configurable is not None and isinstance(o, Configurable):
        return o.config_hash
    elif o is None:
        # backwards compatibility
        return "None"
    else:
        raise TypeError(f"No hashing method for object type {type(o).__name__}")


def hash_model_data(
    model: Optional[sklearn.base.BaseEstimator] = None,
    data: Optional[Any] = None,
    model_hash: Optional[str] = None,
    data_hash: Optional[str] = None,
    **kwargs,
):
    """
    Hash a model and data pair.

    Args:
        model: sklearn model (optional if model_hash provided)
        data: data object (optional if data_hash provided)
        model_hash: pre-computed model hash (optional if model provided)
        data_hash: pre-computed data hash (optional if data provided)
        **kwargs: additional kwargs passed to hash_sklearn

    Returns:
        str: combined hash of model and data
    """
    if model_hash is None:
        if model is None:
            raise ValueError("Either model or model_hash must be provided")
        model_hash = hash_sklearn(model, **kwargs)

    if data_hash is None:
        if data is None:
            raise ValueError("Either data or data_hash must be provided")
        data_hash = hash_data(data)

    return hash_str(f"{data_hash}_{model_hash}")


class EstimatorFactory(ABC):
    @abstractmethod
    def save(self, path: Union[str, Path], model: sklearn.base.BaseEstimator):
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> sklearn.base.BaseEstimator:
        pass


class UnsupportedEstimator(Exception):
    """Raised when an estimator does not support the required save/load methods."""

    pass


class DefaultEstimatorFactory(EstimatorFactory):
    estimator_cls = None

    def __init__(self, save_fn_name: str = "save", load_fn_name: str = "load"):
        self.save_fn_name = save_fn_name
        self.load_fn_name = load_fn_name

    def _get_save_fn(self, model: sklearn.base.BaseEstimator):
        return getattr(model, self.save_fn_name, None)

    def _get_load_fn(self, model_cls: type):
        return getattr(model_cls, self.load_fn_name, None)

    def save(self, path: Union[str, Path], model: sklearn.base.BaseEstimator):
        path = Path(path)
        model_cls = model.__class__
        save_fn = self._get_save_fn(model)
        load_fn = self._get_load_fn(model_cls)
        if save_fn is not None and load_fn is not None:
            self.estimator_cls = model.__class__
            model.save(path)
        else:
            raise UnsupportedEstimator(f"Estimator {model_cls.__name__} does not support required methods '{self.save_fn_name}' and '{self.load_fn_name}'")

    def load(self, path: Union[str, Path]) -> sklearn.base.BaseEstimator:
        path = Path(path)
        if self.estimator_cls is None:
            raise ValueError("Estimator factory doesn't have an estimator class")
        load_fn = self._get_load_fn(self.estimator_cls)
        if load_fn is None:
            raise UnsupportedEstimator(f"Estimator {self.estimator_cls.__name__} does not support required method '{self.load_fn_name}'")
        return load_fn(path)


DEFAULT_FACTORIES: List[DefaultEstimatorFactory] = [
    DefaultEstimatorFactory(save_fn_name="save", load_fn_name="load"),
    DefaultEstimatorFactory(save_fn_name="save_model", load_fn_name="load_model"),
]


@contextlib.contextmanager
def file_lock_context(lock_file: Union[str, Path], timeout: Optional[float] = None):
    """A context manager for file-based locking that cleans up the lock file."""
    if timeout is None:
        timeout = config.FILE_LOCK_TIMEOUT

    lock_path = Path(str(lock_file) + ".lock")
    lock = FileLock(lock_path, timeout=timeout)

    with lock:
        yield


@typeguard.typechecked
def load_model(
    model_path: Union[str, Path],
) -> sklearn.base.BaseEstimator:
    model_path = Path(model_path)
    # we lock the full load process here,
    # so the load implementation of the estimator factory doesn't need to concern itself with threading
    with file_lock_context(model_path):
        model = load_model_pickle(model_path)
        if isinstance(model, EstimatorFactory):
            model = model.load(model_path)
        return model


@typeguard.typechecked
def model_from_params(params: dict, class_name: str, module_name: str) -> sklearn.base.BaseEstimator:
    model_cls = getattr(importlib.import_module(module_name), class_name)
    return model_cls(**params)


@typeguard.typechecked
def load_model_pickle(
    model_path: Path,
) -> Union[sklearn.base.BaseEstimator, EstimatorFactory]:
    model_path = model_path.with_suffix(".pkl")
    return joblib.load(model_path)


@typeguard.typechecked
def store_model(model_path: Union[str, Path], model: sklearn.base.BaseEstimator, overwrite: bool = False):
    model_path = Path(model_path)
    pkl_path = model_path.with_suffix(".pkl")
    if pkl_path.exists() and not overwrite:
        return

    if hasattr(model, "_estimator_factory"):
        model._estimator_factory.save(model_path, model)
        # store factory
        store_model_pickle(model_path, model._estimator_factory, overwrite=overwrite)
        return

    # try default estimator factory
    for model_factory in DEFAULT_FACTORIES:
        try:
            model_factory.save(model_path, model)
            # store factory
            store_model_pickle(model_path, model_factory, overwrite=overwrite)
            return
        except UnsupportedEstimator:
            continue
        except Exception as factory_exception:
            # Try default pickle before re-raising the error
            try:
                store_model_pickle(model_path, model, overwrite=overwrite)
                return
            except Exception as pickle_exception:
                # Re-raise the original exception from the factory
                raise pickle_exception from factory_exception
    # finally default to pickle
    store_model_pickle(model_path, model, overwrite=overwrite)


@typeguard.typechecked
def store_model_pickle(
    model_path: Union[str, Path],
    model: Union[sklearn.base.BaseEstimator, EstimatorFactory],
    overwrite: bool = False,
):
    model_path = model_path.with_suffix(".pkl")
    if model_path.exists() and not overwrite:
        return

    with file_lock_context(model_path):
        joblib.dump(model, model_path)


def save_json(data_path: Union[str, Path], data: dict):
    with open(data_path, "w") as f:
        json.dump(data, f)


def load_json(data_path: Union[str, Path]) -> dict:
    with open(data_path, "r") as f:
        return json.load(f)


SUPPORTED_DATA_TYPES = {
    "numpy": {
        "cls": np.ndarray,
        "extension": ".npy",
        "save_fn": partial(np.save, allow_pickle=True),
        "load_fn": partial(np.load, allow_pickle=True),
    },
    "json": {
        "cls": dict,
        "extension": ".json",
        "save_fn": save_json,
        "load_fn": load_json,
    },
    "data_arguments": {
        "cls": DataArguments,
        "extension": ".data_args",
        "save_fn": lambda data_path, data: data.save(data_path),
        "load_fn": DataArguments.load,
    },
}

# Add config_dataclass support if available
if Configurable is not None:
    SUPPORTED_DATA_TYPES["config_dataclass"] = {
        "cls": Configurable,
        "extension": ".configurable",
        "save_fn": lambda data_path, data: data.save(data_path),
        "load_fn": Configurable.load,
    }

if torch is not None:
    SUPPORTED_DATA_TYPES["torch"] = {
        "cls": torch.Tensor,
        "extension": ".pt",
        "save_fn": torch.save,
        "load_fn": torch.load,
    }


@typeguard.typechecked
def store_data(
    data_path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
):
    data_path = Path(data_path)

    for _, data_type_info in SUPPORTED_DATA_TYPES.items():
        if isinstance(data, data_type_info["cls"]):
            data_path = data_path.with_suffix(data_type_info["extension"])
            if data_path.exists() and not overwrite:
                # if the file exists and we're not overwriting, return
                return
            with file_lock_context(data_path):
                data_type_info["save_fn"](data_path, data)
            return

    # If no supported type found
    supported_types = [info["cls"].__name__ for info in SUPPORTED_DATA_TYPES.values()]
    raise TypeError(f"Data type not supported, got {type(data).__name__}. Currently supported types: {', '.join(supported_types)}")


@typeguard.typechecked
def load_data(data_path: Union[str, Path]) -> Any:
    data_path = Path(data_path)

    # If data_path already has a suffix, try to load directly
    if data_path.suffix:
        for data_type_info in SUPPORTED_DATA_TYPES.values():
            if data_path.suffix == data_type_info["extension"]:
                with file_lock_context(data_path):
                    return data_type_info["load_fn"](data_path)
        raise ValueError(f"Unsupported file extension: {data_path.suffix}")

    # If no suffix, try to find a file with any supported extension
    for data_type_info in SUPPORTED_DATA_TYPES.values():
        candidate_path = data_path.with_suffix(data_type_info["extension"])
        if candidate_path.exists():
            with file_lock_context(candidate_path):
                return data_type_info["load_fn"](candidate_path)

    # If no file found with any supported extension
    supported_extensions = [info["extension"] for info in SUPPORTED_DATA_TYPES.values()]
    raise FileNotFoundError(f"No data file found with supported extensions {supported_extensions} for path: {data_path}")


# A dummy context manager that does nothing, mimicking tqdm's interface.
class NoOpTqdm:
    """A no-op tqdm look-alike for when the progress bar is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, n=1):
        pass

    def set_postfix(self, ordered_dict=None, **kwargs):
        pass
