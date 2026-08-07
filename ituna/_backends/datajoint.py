from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union
import warnings

import sklearn
import typeguard

from ituna import estimator
from ituna._backends import base
from ituna._backends import utils
from ituna._cache_guard import suspend_global_cache_patch

_DATAJOINT_IMPORTS_MISSING = []
try:
    import datajoint as dj
    from dj_ml_core import fields as db_fields
    from dj_ml_core import login
    from dj_ml_core.core import Schema
except ImportError:
    _DATAJOINT_IMPORTS_MISSING = ["datajoint", "dj_ml_core"]

_DATAJOINT_AVAILABLE = len(_DATAJOINT_IMPORTS_MISSING) == 0

if not _DATAJOINT_AVAILABLE:

    class DatajointBackend:
        """Stub when datajoint or dj_ml_core is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The following packages are required for DatajointBackend: "
                f"{', '.join(_DATAJOINT_IMPORTS_MISSING)}. "
                "Please install them with: pip install ituna[datajoint]. "
                "Note: dj_ml_core may need to be installed from the project's wheel (e.g. third_party/dj_ml_core-*.whl)."
            )

    def get_sweep_jobs_for_table(*args, **kwargs):
        raise ImportError(
            "The following packages are required for get_sweep_jobs_for_table: "
            f"{', '.join(_DATAJOINT_IMPORTS_MISSING)}. "
            "Please install them with: pip install ituna[datajoint]. "
            "Note: dj_ml_core may need to be installed from the project's wheel (e.g. third_party/dj_ml_core-*.whl)."
        )

else:

    @typeguard.typechecked
    class DatajointBackend(base.Backend, base.DistributedComputationMixin):
        """A backend that uses DataJoint for storing models and training jobs.

        This backend provides a robust and scalable solution for managing and
        tracking machine learning experiments by leveraging a formal data model
        and a relational database. It supports distributed computation, allowing
        models to be trained in parallel across multiple workers.

        Attributes
        ----------
        schema : dj.Schema
            The DataJoint schema object.
        """

        def __init__(
            self,
            cache_dir: Union[str, Path],
            host: Optional[str] = None,
            user: Optional[str] = None,
            password: Optional[str] = None,
            schema_name: Optional[str] = None,
            trigger_type: str = "auto",
            num_workers: Optional[int] = 1,
            progress_bar: bool = True,
            sweep_type: str = "uuid",
            sweep_name: Optional[str] = None,
            cache_refresh_interval: float = 1.0,
            fit_time_out: Optional[float] = None,
            verbose_login: bool = False,
        ):
            """Initialize the DataJoint backend.

            Parameters
            ----------
            cache_dir : str or Path
                The directory to use for caching DataJoint models and data.
            host : str, optional
                The database host. If not provided, it is read from the
                environment variables.
            user : str, optional
                The database user. If not provided, it is read from the
                environment variables.
            password : str, optional
                The database password. If not provided, it is read from the
                environment variables.
            schema_name : str, optional
                The name of the database schema. If not provided, it is read from
                the environment variables.
            trigger_type : {'auto', 'manual'}
                The trigger type for starting the sweep. 'auto' starts worker
                processes automatically, while 'manual' requires the user to start
                them.
            num_workers : int, optional
                The number of worker processes to start when `trigger_type` is
                'auto'. Required if `trigger_type` is 'auto'.
            progress_bar : bool
                Whether to display a progress bar while waiting for the sweep to
                complete.
            sweep_type : {'uuid', 'constant'}
                The type of sweep name to generate. 'uuid' generates a unique sweep_name`.
            sweep_name : str, optional
                The name of the sweep to use when `sweep_type` is 'constant'.
                Required if `sweep_type` is 'constant'.
            cache_refresh_interval : float
                The interval in seconds at which to check the status of the sweep.
            fit_time_out : float, optional
                The maximum time in seconds to wait for the sweep to complete.
            """
            super().__init__(
                trigger_type=trigger_type,
                num_workers=num_workers,
                progress_bar=progress_bar,
                sweep_type=sweep_type,
                sweep_name=sweep_name,
                cache_refresh_interval=cache_refresh_interval,
                fit_time_out=fit_time_out,
            )
            self.host = host
            self.user = user
            self.password = password
            self.schema_name = schema_name

            # try connecting to datajoint
            env_vars = login.connect_to_database(
                host=self.host,
                user=self.user,
                password=self.password,
                verbose=verbose_login,
            )
            if self.schema_name is None:
                self.schema_name = env_vars["DATAJOINT_SCHEMA_NAME"]

            if self.schema_name is None:
                raise ValueError("schema_name must be specified directly or via env variable DATAJOINT_SCHEMA_NAME")

            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._worker_log_dir = self.datajoint_cache_dir / "worker_logs"

            self._model_cache_dir.mkdir(parents=True, exist_ok=True)
            self._data_cache_dir.mkdir(parents=True, exist_ok=True)
            self._trained_models_dir.mkdir(parents=True, exist_ok=True)
            self._worker_log_dir.mkdir(parents=True, exist_ok=True)

            self.schema, self._tables = self._create_tables()

        @property
        def datajoint_cache_dir(self) -> Path:
            return self.cache_dir / "datajoint" / f"{self.schema_name}"

        def __getstate__(self):
            """Customize object serialization for pickling.

            This method excludes the DataJoint schema and table objects, which are
            not picklable as they hold database connections. These objects will be
            recreated during unpickling.
            """
            state = self.__dict__.copy()
            # Don't pickle schema and tables, as they hold database connections
            # and are not picklable. They will be recreated in __setstate__.
            state.pop("schema", None)
            state.pop("_tables", None)
            return state

        def __setstate__(self, state):
            """Customize object deserialization for unpickling.

            This method restores the object's state and then re-initializes the
            backend to re-establish the database connection and recreate the schema
            and table objects.
            """
            self.__dict__.update(state)
            # Re-initialize the backend to re-create schema and tables
            # The original init arguments are stored as attributes, so we can just
            # call __init__ again.
            self.__init__(
                cache_dir=self.cache_dir,
                host=self.host,
                user=self.user,
                password=self.password,
                schema_name=self.schema_name,
                trigger_type=self.trigger_type,
                num_workers=self.num_workers,
                progress_bar=self.progress_bar,
                sweep_type=self.sweep_type,
                sweep_name=self.sweep_name,
                cache_refresh_interval=self.cache_refresh_interval,
                fit_time_out=self.fit_time_out,
            )

        @property
        def _model_cache_dir(self) -> Path:
            return self.datajoint_cache_dir / "models"

        @property
        def _data_cache_dir(self) -> Path:
            return self.datajoint_cache_dir / "data"

        @property
        def _trained_models_dir(self) -> Path:
            return self.datajoint_cache_dir / "trained_models"

        @property
        def tables(self):
            return self._tables

        def _create_tables(self):
            """Create the DataJoint tables for the backend."""
            schema = Schema(self.schema_name, context=locals(), create_tables=True, create_schema=True)
            backend_self = self

            class ModelTable(dj.Manual):
                model_hash = db_fields.CharField(length=128, primary_key=True)
                model_path = db_fields.CharField(length=128, default="null")
                model_config = db_fields.JSONField()
                model_class_name = db_fields.CharField(length=128, default="null")
                model_module_name = db_fields.CharField(length=128, default="null")

                def insert_model(self, model: sklearn.base.BaseEstimator, skip_duplicates=True) -> str:
                    """Insert a model into the table.

                    Parameters
                    ----------
                    model : sklearn.base.BaseEstimator
                        The model to insert.
                    skip_duplicates : bool
                        If True, skip insertion if the model already exists.

                    Returns
                    -------
                    str
                        The hash of the inserted model.
                    """
                    model = sklearn.base.clone(model, safe=False)
                    cls = model.__class__
                    class_name = cls.__name__
                    module_name = cls.__module__

                    # in the ModelTable we only store models that are unique up to their random_state
                    # so we need to reset random_state to None if model is deterministic
                    if not estimator.check_non_deterministic(model) and "random_state" in model.get_params():
                        model.set_params(random_state=None)

                    model_hash = utils.hash_model(model)
                    config = model.get_params()

                    # test whether model can be created from class and params alone
                    try:
                        model_from_params = utils.model_from_params(config, class_name, module_name)
                        recreated_hash = utils.hash_model(sklearn.base.clone(model_from_params, safe=False))
                        assert model_hash == recreated_hash, "Model cannot be re-created from params alone"
                    except Exception as e:
                        model_from_params = None

                        warnings.warn(f"Model {class_name} from {module_name} cannot be re-created from params: {e}, falling back to storing model")

                    if model_from_params is None:
                        model_path_abs = backend_self._model_cache_dir / model_hash
                        utils.store_model(model_path_abs, model)
                        model_path = model_path_abs.relative_to(backend_self.datajoint_cache_dir)
                    else:
                        model_path = None

                    for key in config.keys():
                        if isinstance(config[key], sklearn.base.BaseEstimator):
                            config[key] = dict(
                                key=dict(model_hash=self.insert_model(config[key])),
                                table=self.table_name,
                            )
                    self.insert1(
                        dict(
                            model_hash=model_hash,
                            model_path=str(model_path) if model_path else None,
                            model_config=config,
                            model_class_name=class_name,
                            model_module_name=module_name,
                        ),
                        skip_duplicates=skip_duplicates,
                    )

                    return model_hash

                def get_model(self, key) -> sklearn.base.BaseEstimator:
                    """Retrieve a model from the table.

                    Parameters
                    ----------
                    key : dict
                        The primary key of the model to retrieve.

                    Returns
                    -------
                    sklearn.base.BaseEstimator
                        The reconstructed model.
                    """
                    config_inst = self
                    if key is not None:
                        config_inst = config_inst & key

                    row = config_inst.fetch1()

                    params = row["model_config"]

                    for key in params.keys():
                        if isinstance(params[key], dict) and "key" in params[key] and "table" in params[key]:
                            if params[key]["table"] == self.table_name:
                                params[key] = self.get_model(params[key]["key"])
                            else:
                                raise ValueError(f"Table {params[key]['table']} not supported for retrieving model")

                    class_name = row["model_class_name"]
                    module_name = row["model_module_name"]
                    model_path = row["model_path"]

                    if model_path is not None and model_path != "null":
                        model_path_abs = backend_self.datajoint_cache_dir / model_path
                        return utils.load_model(model_path_abs)
                    else:
                        return utils.model_from_params(params=params, class_name=class_name, module_name=module_name)

            class DatasetTable(dj.Manual):
                data_hash = db_fields.CharField(length=128, primary_key=True)
                data_path = db_fields.CharField(length=128, default="null")
                data_class_name = db_fields.CharField(length=128, default="null")
                data_module_name = db_fields.CharField(length=128, default="null")
                data_config = db_fields.JSONField()
                table_reference = db_fields.JSONField()
                data_dependencies = db_fields.JSONField()
                stats = db_fields.JSONField()

                def _insert_data(
                    self,
                    data_hash: str,
                    data_object: Any,
                    data_path: Optional[str] = None,
                    data_dependencies: Optional[dict] = None,
                    skip_duplicates=True,
                ):
                    """
                    Compute stats + config dict and insert into the table.
                    """

                    def get_stats(data_item):
                        """Get statistics for a data item."""
                        stats_dict = {}
                        attributes = ["shape", "dtype", "__len__", "size"]
                        to_string = ["dtype"]
                        for attribute in attributes:
                            if hasattr(data_item, attribute):
                                stats_dict[attribute] = getattr(data_item, attribute)
                                if attribute in to_string:
                                    stats_dict[attribute] = str(stats_dict[attribute])
                                if callable(stats_dict[attribute]):
                                    try:
                                        stats_dict[attribute] = stats_dict[attribute]()
                                    except Exception:
                                        del stats_dict[attribute]

                        return stats_dict

                    def get_config_dicts(data_item):
                        # check for Configurable class properties
                        if hasattr(data_item, "config_hash") and hasattr(data_item, "to_dict"):
                            return data_item.to_dict()
                        return None

                    def create_table_reference(data_item):
                        """Resolve reference to another datajoint table."""
                        if hasattr(data_item, "_datajoint_key") and hasattr(data_item, "_datajoint_table"):
                            return dict(
                                key=data_item._datajoint_key,
                                table=data_item._datajoint_table,
                            )
                        return None

                    self.insert1(
                        dict(
                            data_hash=data_hash,
                            data_path=str(data_path) if data_path else None,
                            data_class_name=str(data_object.__class__.__name__),
                            data_module_name=str(data_object.__class__.__module__),
                            stats=get_stats(data_object),
                            data_config=get_config_dicts(data_object),
                            table_reference=create_table_reference(data_object),
                            data_dependencies=data_dependencies,
                        ),
                        skip_duplicates=skip_duplicates,
                    )

                def insert_data(self, data: utils.DataArguments, skip_duplicates=True) -> str:
                    """Insert a dataset into the table.

                    Parameters
                    ----------
                    data : dict
                        The dataset to insert, expected to contain 'X' and 'y'.
                    skip_duplicates : bool
                        If True, skip insertion if the dataset already exists.

                    Returns
                    -------
                    str
                        The hash of the inserted dataset.
                    """

                    data_objects_paths = data.save_data_objects(root_dir=backend_self._data_cache_dir)

                    for data_object_hash, data_object_path in data_objects_paths.items():
                        data_object = data.data_objects[data_object_hash]
                        rel_path = data_object_path.relative_to(backend_self.datajoint_cache_dir) if data_object_path else None
                        self._insert_data(
                            data_hash=data_object_hash,
                            data_path=rel_path,
                            data_object=data_object,
                            skip_duplicates=skip_duplicates,
                        )

                    self._insert_data(
                        data_hash=data.hash,
                        data_path=None,
                        data_object=data,
                        data_dependencies=data.hash_dict,
                        skip_duplicates=skip_duplicates,
                    )

                    return data.hash

                def get_data(self, key) -> dict:
                    """Retrieve a dataset from the table.

                    Parameters
                    ----------
                    key : dict
                        The primary key of the dataset to retrieve.

                    Returns
                    -------
                    dict
                        The dataset.
                    """
                    dataset_inst = self
                    if key is not None:
                        dataset_inst = dataset_inst & key
                    row = dataset_inst.fetch1()

                    data_hash = row["data_hash"]

                    # check if data_class_type is "DataArguments"
                    if row["data_class_name"] == "DataArguments":
                        data = utils.DataArguments.from_hash_dict(
                            hash_dict=row["data_dependencies"],
                            root_dir=backend_self._data_cache_dir,
                            load_fn=lambda value_hash: self.get_data(dict(data_hash=value_hash)),
                        )
                    elif row["table_reference"] is not None:
                        table_name = row["table_reference"]["table"]
                        if table_name == backend_self.tables.TrainedModelsTable().table_name:
                            data = backend_self.tables.TrainedModelsTable().get_trained_model(row["table_reference"]["key"])
                        else:
                            raise ValueError(f"Table {table_name} not supported")
                    else:
                        data_path_abs = backend_self.datajoint_cache_dir / row["data_path"]
                        data = utils.load_data(data_path_abs)

                    hash_again = utils.hash_data(data)
                    if hash_again != data_hash:
                        raise ValueError("Data hash does not match, data may have been corrupted")

                    return data

            class ModelTrainingTable(dj.Manual):
                arg_hash = db_fields.CharField(length=128, primary_key=True)
                data_hash = db_fields.TableField(
                    table=DatasetTable,
                )
                model_hash = db_fields.TableField(
                    table=ModelTable,
                )
                model_seed = db_fields.IntField(comment="Index or Random Seed of the model")

                def insert_model_training(
                    self,
                    model_seed: Optional[int] = None,
                    model_hash: Optional[str] = None,
                    data_hash: Optional[str] = None,
                    model: Optional[sklearn.base.BaseEstimator] = None,
                    data: Optional[utils.DataArguments] = None,
                    sweep_name: Optional[str] = None,
                    skip_duplicates: bool = True,
                    **kwargs,
                ):
                    """Insert a model training entry into the table.

                    This defines a unique combination of a model, a dataset, and
                    a seed, which constitutes a single training job.

                    Parameters
                    ----------
                    model_seed : int
                        The seed for the model's random state.
                    model_hash : str, optional
                        The hash of a model already in the `ModelTable`.
                    data_hash : str, optional
                        The hash of input data already in the `DatasetTable`.
                    model : sklearn.base.BaseEstimator, optional
                        A model instance. If provided, it will be inserted into
                        the `ModelTable`.
                    data : utils.DataArguments, optional
                        Input data for model.fit() If provided, it will be inserted into the
                        `DatasetTable`.
                    sweep_name : str, optional
                        If provided, this training run will be associated with the
                        given sweep.
                    skip_duplicates : bool
                        If True, skip insertion if the model training entry already exists.
                    **kwargs : dict
                        Additional keyword arguments for the insert call.
                    """
                    assert model_seed != -1, "model_seed -1 is reserved to encode None/null entries"
                    if model_hash is None:
                        model_hash = ModelTable().insert_model(model, skip_duplicates=skip_duplicates)
                    if data_hash is None:
                        data_hash = DatasetTable().insert_data(data, skip_duplicates=skip_duplicates)

                    non_deterministic = estimator.check_non_deterministic(model)
                    if not non_deterministic:
                        # use model.random_state as model_seed instead
                        model_seed = model.get_params().get("random_state", model_seed)

                    if model_seed is None:
                        # integer type of mysql does not support None or null
                        # int "null"s get converted to 0
                        # 0 may be a valid model seed, so we use -1 instead
                        model_seed = -1
                    args_dict = dict(
                        model_seed=model_seed,
                        model_hash=model_hash,
                        data_hash=data_hash,
                    )

                    arg_hash = utils.hash_dict(args_dict)
                    self.insert1(
                        dict(
                            arg_hash=arg_hash,
                            **args_dict,
                        ),
                        skip_duplicates=skip_duplicates,
                        **kwargs,
                    )

                    if sweep_name is not None:
                        SweepRunsTable().insert1(
                            dict(
                                sweep_name=sweep_name,
                                arg_hash=arg_hash,
                            ),
                            skip_duplicates=True,
                            **kwargs,
                        )

                    return arg_hash

                def get_model_training(
                    self,
                    key,
                ) -> Tuple[
                    sklearn.base.BaseEstimator,
                    utils.DataArguments,
                    int,
                ]:
                    """Retrieve the components for a model training run.

                    Parameters
                    ----------
                    key : dict
                        The primary key of the training run.

                    Returns
                    -------
                    tuple
                        A tuple containing the model, the data, and the seed.
                    """
                    model_training_inst = self
                    if key is not None:
                        model_training_inst = model_training_inst & key
                    assert len(model_training_inst) == 1, "Expected a single row"
                    row = model_training_inst.fetch1()

                    model_hash = row["model_hash"]
                    data_hash = row["data_hash"]
                    model_seed = row["model_seed"]
                    if model_seed == -1:
                        model_seed = None

                    model = ModelTable().get_model(dict(model_hash=model_hash))
                    data = DatasetTable().get_data(dict(data_hash=data_hash))

                    model = estimator.clone_with_seed(model, model_seed)

                    return model, data, model_seed

            class SweepTable(dj.Manual):
                sweep_name = db_fields.CharField(length=128, primary_key=True)

            class SweepRunsTable(dj.Manual):
                sweep_name = db_fields.TableField(table=SweepTable, primary_key=True)
                arg_hash = db_fields.TableField(table=ModelTrainingTable, primary_key=True)

                def insert1(self, row, **kwargs):
                    # automatically create the sweep if it doesn't exist
                    SweepTable().insert1(
                        dict(sweep_name=row["sweep_name"]),
                        skip_duplicates=True,
                    )
                    super().insert1(row, **kwargs)

            class TrainedModelsTable(dj.Computed):
                arg_hash = db_fields.TableField(table=ModelTrainingTable, primary_key=True)
                log_dir = db_fields.CharField(length=128)

                def make(self, key, **kwargs):
                    """Train a single model.

                    This method is called by DataJoint's `populate` process.

                    Parameters
                    ----------
                    key : dict
                        The primary key from the `ModelTrainingTable`.
                    **kwargs : dict
                        Additional arguments for the training process.
                    """
                    print(f"Start training {key}", flush=True)
                    print(f"\tkwargs: {kwargs}", flush=True)

                    # get model training entry
                    model, data_args, _ = ModelTrainingTable().get_model_training(key)

                    with suspend_global_cache_patch():
                        trained_model = model.fit(*data_args.args, **data_args.kwargs)

                    trained_model_logs_abs = backend_self._trained_models_dir / f"{key['arg_hash']}"
                    print(
                        f"Storing trained model in {trained_model_logs_abs}",
                        flush=True,
                    )
                    utils.store_model(model_path=trained_model_logs_abs, model=trained_model)
                    trained_model_logs = trained_model_logs_abs.relative_to(backend_self.datajoint_cache_dir)

                    insert_key = key.copy()
                    insert_key["log_dir"] = str(trained_model_logs)
                    self.insert1(insert_key)
                    return key

                def get_trained_model(self, key) -> sklearn.base.BaseEstimator:
                    """Retrieve a trained model from the cache.

                    Parameters
                    ----------
                    key : dict
                        The primary key of the trained model.

                    Returns
                    -------
                    sklearn.base.BaseEstimator
                        The trained model.
                    """
                    inst = self
                    if key is not None:
                        inst = inst & key

                    row = inst.fetch1()
                    log_dir_abs = backend_self.datajoint_cache_dir / row["log_dir"]
                    model = utils.load_model(log_dir_abs)
                    # add arg_hash to model for later reference
                    model._datajoint_key = key
                    model._datajoint_table = self.table_name
                    return model

            class EncoderTransformsTable(dj.Manual):
                arg_hash = db_fields.CharField(length=128, primary_key=True)
                trained_model_hash = db_fields.ProjectedTableField(
                    table=TrainedModelsTable,
                    attribute_name="trained_model_hash",
                    key_name="arg_hash",
                )
                data_hash = db_fields.TableField(
                    table=DatasetTable,
                )

                def insert_encoder_transform(
                    self,
                    trained_model_hash: str,
                    data_hash: Optional[str] = None,
                    data: Optional[dict] = None,
                    skip_duplicates: bool = True,
                    **kwargs,
                ):
                    if data_hash is None:
                        data_hash = DatasetTable().insert_data(data, skip_duplicates=skip_duplicates)

                    arg_hash = utils.hash_dict(
                        dict(
                            trained_model_hash=trained_model_hash,
                            data_hash=data_hash,
                        )
                    )

                    self.insert1(
                        dict(
                            arg_hash=arg_hash,
                            trained_model_hash=trained_model_hash,
                            data_hash=data_hash,
                        ),
                        skip_duplicates=skip_duplicates,
                        **kwargs,
                    )

                def get_encoder_transform(self, key) -> Tuple[sklearn.base.BaseEstimator, dict]:
                    """Retrieve an encoder transform from the cache."""
                    inst = self
                    if key is not None:
                        inst = inst & key
                    row = inst.fetch1()

                    trained_model_hash = row["trained_model_hash"]
                    data_hash = row["data_hash"]

                    trained_model = TrainedModelsTable().get_trained_model(dict(trained_model_hash=trained_model_hash))
                    data = DatasetTable().get_data(dict(data_hash=data_hash))

                    return trained_model, data

            class EncoderTransformsResultsTable(dj.Computed):
                arg_hash = db_fields.TableField(table=EncoderTransformsTable, primary_key=True)
                result_data_hash = db_fields.TableField(table=DatasetTable)

                def make(self, key, **kwargs):
                    """Compute the encoder transform results."""
                    trained_model, data = EncoderTransformsTable().get_encoder_transform(key)
                    with suspend_global_cache_patch():
                        result = trained_model.transform(data)
                    result_data_hash = DatasetTable().insert_data(result, skip_duplicates=True)
                    self.insert1(
                        dict(
                            arg_hash=key["arg_hash"],
                            result_data_hash=result_data_hash,
                        )
                    )
                    return key

            return (
                schema,
                SimpleNamespace(
                    ModelTable=schema(ModelTable, context=locals()),
                    DatasetTable=schema(DatasetTable, context=locals()),
                    ModelTrainingTable=schema(ModelTrainingTable, context=locals()),
                    SweepTable=schema(SweepTable, context=locals()),
                    SweepRunsTable=schema(SweepRunsTable, context=locals()),
                    TrainedModelsTable=schema(TrainedModelsTable, context=locals()),
                    EncoderTransformsTable=schema(EncoderTransformsTable, context=locals()),
                    EncoderTransformsResultsTable=schema(EncoderTransformsResultsTable, context=locals()),
                ),
            )

        def _get_fit_distributed_cmd(self, sweep_name: str):
            """
            Get the command to run the fit_distributed script.
            """
            cmd = [
                "ituna-fit-distributed-datajoint",
                "--sweep-name",
                sweep_name,
            ]
            if self.schema_name:
                cmd.extend(["--schema-name", self.schema_name])
            if self.host:
                cmd.extend(["--host", self.host])
            if self.user:
                cmd.extend(["--user", self.user])
            if self.password:
                cmd.extend(["--password", self.password])
            if self.cache_dir:
                cmd.extend(["--cache-dir", str(self.cache_dir.resolve())])
            return cmd

        def _retrieve_trained_model(self, model_data_hash: str) -> Optional[sklearn.base.BaseEstimator]:
            """
            Retrieve a trained model from datajoint.
            """
            try:
                return self.tables.TrainedModelsTable().get_trained_model(dict(arg_hash=model_data_hash))
            except dj.DataJointError:
                return None

        def get_sweeps(
            self,
            sweep_names: Union[str, List[str]],
            partial_match: bool = False,
        ) -> dj.expression.QueryExpression:
            """Get a query for sweeps matching the given names.

            Parameters
            ----------
            sweep_names : str or list of str
                The name(s) of the sweep(s) to retrieve.
            partial_match : bool
                If True, `sweep_names` are treated as patterns to match against.

            Returns
            -------
            dj.QueryExpression
                A DataJoint query expression for the matching sweeps.
            """
            if isinstance(sweep_names, str):
                sweep_names = [sweep_names]
            sweep_restriction = [f"sweep_name like '%{sweep_name}%'" if partial_match else f"sweep_name = '{sweep_name}'" for sweep_name in sweep_names]

            return self.tables.SweepTable() & sweep_restriction

        def _get_sweep_status(
            self,
            sweep_names: Union[str, List[str]],
            models_of_interest: Optional[List[str]] = None,
        ):
            """
            Get the status of a sweep.
            """
            sweeps = self.get_sweeps(sweep_names)

            sweep_runs = self.tables.SweepRunsTable() & sweeps

            queued_models = self.tables.ModelTrainingTable() & sweep_runs

            trained_models = self.tables.TrainedModelsTable() & queued_models

            jobs_stats = get_sweep_jobs_for_table(
                restrictions=queued_models,
                table=self.tables.TrainedModelsTable,
                key_table=self.tables.ModelTrainingTable,
                schema=self.schema,
            )

            status = {}
            status["sweep_total"] = len(queued_models)
            status["sweep_trained"] = len(trained_models)
            status["sweep_errors"] = jobs_stats["error_count"]
            status["sweep_reserved"] = jobs_stats["reserved_count"]
            status["sweep_completed"] = status["sweep_trained"] + status["sweep_errors"]

            # Calculate status for models of interest if provided
            if models_of_interest:
                model_restrictions = [dict(arg_hash=model_hash) for model_hash in models_of_interest]
                model_job_stats = get_sweep_jobs_for_table(
                    restrictions=model_restrictions,
                    table=self.tables.TrainedModelsTable,
                    key_table=self.tables.ModelTrainingTable,
                    schema=self.schema,
                )
                status["total"] = len(queued_models & model_restrictions)
                status["trained"] = len(trained_models & model_restrictions)
                status["errors"] = model_job_stats["error_count"]
                status["reserved"] = model_job_stats["reserved_count"]
                status["completed"] = status["trained"] + status["errors"]

            return status

        @typeguard.typechecked
        def fit_models(
            self,
            models: List[sklearn.base.BaseEstimator],
            *args,
            **kwargs,
        ) -> List[Optional[sklearn.base.BaseEstimator]]:
            """
            Fit models in the queue.
            """
            data = utils.DataArguments(*args, **kwargs)
            sweep_name = self._get_sweep_name()

            training_queue = []
            for i, model in enumerate(models):
                model_training_hash = self.tables.ModelTrainingTable().insert_model_training(
                    model_seed=i,
                    model=model,
                    data=data,
                    sweep_name=sweep_name,
                    skip_duplicates=True,
                )
                training_queue.append(model_training_hash)

            self._trigger_sweep(sweep_name)

            # wait for sweep to finish
            trained_models = self._collect_trained_models(sweep_name, training_queue)

            return trained_models

        def fit_sweep_models(
            self,
            sweep_name: Union[str, List[str]],
            limit: Optional[int] = None,
            order: str = "random",
            suppress_errors: bool = False,
            partial_match: bool = False,
            make_kwargs: Optional[dict] = None,
        ):
            """Populate the TrainedModelsTable for a given sweep.

            This method triggers the training of all models associated with a
            sweep that have not yet been trained.

            Parameters
            ----------
            sweep_name : str or list of str
                The UUID(s) of the sweep(s) to process.
            limit : int, optional
                The maximum number of models to train.
            order : {'original', 'reverse', 'random'}
                The order in which to process the training jobs.
            suppress_errors : bool
                If True, suppress errors during training and continue.
            partial_match : bool
                If True, `sweep_name` is treated as a pattern.
            make_kwargs : dict, optional
                Keyword arguments to pass to the `make` method of the
                `TrainedModelsTable`.
            """
            sweeps = self.get_sweeps(sweep_name, partial_match=partial_match)
            if not len(sweeps):
                warnings.warn(f"No sweeps found for UUID(s): {sweep_name}")
                return

            sweep_runs = self.tables.SweepRunsTable() & sweeps
            restriction = self.tables.ModelTrainingTable() & sweep_runs

            if not restriction:
                warnings.warn(f"No models to train for sweep(s): {sweep_name}")
                return

            print(f"Populating {len(restriction)} models for sweep {sweep_name}")

            errors = self.tables.TrainedModelsTable().populate(
                restriction,
                reserve_jobs=True,
                suppress_errors=suppress_errors,
                order=order,
                max_calls=limit,
                display_progress=True,
                make_kwargs=make_kwargs,
            )

            if errors is not None and len(errors) > 0:
                for error in errors:
                    print(error, flush=True)
                    traceback.print_exc()
                print("Showed errors", flush=True)
            else:
                print("Successfully populated Model", flush=True)

        def reset_jobs(
            self,
            sweep_names: Union[str, List[str]],
            partial_match: bool = False,
            error=False,
            reserved=False,
        ):
            """
            Reset the jobs for a given sweep.
            """
            sweeps = self.get_sweeps(sweep_names, partial_match=partial_match)
            if not len(sweeps):
                warnings.warn(f"No sweeps found for UUID(s): {sweep_names}")
                return

            print("Resetting jobs for sweep(s):", sweep_names)

            sweep_runs = self.tables.SweepRunsTable() & sweeps

            queued_models = self.tables.ModelTrainingTable() & sweep_runs

            jobs_stats = get_sweep_jobs_for_table(
                restrictions=queued_models,
                table=self.tables.TrainedModelsTable,
                key_table=self.tables.ModelTrainingTable,
                schema=self.schema,
            )

            if error:
                print(f"Found {jobs_stats['error_count']} error jobs")
                if jobs_stats["error_count"] > 0:
                    confirm = input(f"Are you sure you want to delete {jobs_stats['error_count']} error jobs? (y/N): ")
                    if confirm.lower() in ["y", "yes"]:
                        jobs_stats["error_jobs"].delete()
                        print("Error jobs deleted.")
                    else:
                        print("Error job deletion cancelled.")

            if reserved:
                print(f"Found {jobs_stats['reserved_count']} reserved jobs")
                if jobs_stats["reserved_count"] > 0:
                    confirm = input(f"Are you sure you want to delete {jobs_stats['reserved_count']} reserved jobs? (y/N): ")
                    if confirm.lower() in ["y", "yes"]:
                        jobs_stats["reserved_jobs"].delete()
                        print("Reserved jobs deleted.")
                    else:
                        print("Reserved job deletion cancelled.")

    def get_sweep_jobs_for_table(
        restrictions,
        table: dj.Computed,
        key_table: dj.Manual,
        schema: dj.Schema,
    ):
        """
        Adapted from jobs_helper.ipynb - get jobs for a specific table and sweep.

        Args:
            restrictions: Experiment configurations for the sweep
            table: The table to check jobs for (e.g., ExperimentTable)
            key_table: The table to get keys from (e.g., ExperimentConfigTable)

        Returns:
            dict: {"error_jobs": DataJoint table, "reserved_jobs": DataJoint table,
                   "error_count": int, "reserved_count": int, "table_name": str}
        """
        try:
            table_name = table.table_name

            # Get all error and reserved jobs for this table
            error_jobs = schema.jobs & [
                dict(table_name=table_name, status="error"),
            ]
            reserved_jobs = schema.jobs & [
                dict(table_name=table_name, status="reserved"),
            ]

            # Get arg_hashes for the sweep
            exp_arg_hashes = restrictions.fetch("arg_hash")

            # Get key_hashes for jobs related to this sweep
            table_key_hashes = [dj.key_hash(key) for key in key_table.fetch("KEY") if key["arg_hash"] in exp_arg_hashes]

            if not table_key_hashes:
                return {
                    "error_jobs": error_jobs & [],
                    "reserved_jobs": reserved_jobs & [],
                    "error_count": 0,
                    "reserved_count": 0,
                    "table_name": table_name,
                }

            # Create job restrictions
            job_restrictions = [dict(key_hash=t_key_hash) for t_key_hash in table_key_hashes]

            # Filter jobs for this sweep
            sweep_error_jobs = error_jobs & job_restrictions
            sweep_reserved_jobs = reserved_jobs & job_restrictions

            return {
                "error_jobs": sweep_error_jobs,
                "reserved_jobs": sweep_reserved_jobs,
                "error_count": len(sweep_error_jobs),
                "reserved_count": len(sweep_reserved_jobs),
                "table_name": table_name,
            }

        except Exception as e:
            return {
                "error_jobs": None,
                "reserved_jobs": None,
                "error_count": 0,
                "reserved_count": 0,
                "table_name": getattr(table, "table_name", "unknown"),
                "error": str(e),
            }
