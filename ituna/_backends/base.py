from abc import ABC
from abc import abstractmethod
import subprocess
import time
from typing import List, Optional, Union
import uuid

import sklearn.base
from tqdm import tqdm
import typeguard

from ituna._backends import utils


class Backend(ABC):
    """Abstract base class for a backend.

    A backend is responsible for fitting models and storing the trained models.
    """

    @abstractmethod
    def fit_models(
        self,
        models: List[sklearn.base.BaseEstimator],
        X,
        y=None,
        **fit_params,
    ) -> List[sklearn.base.BaseEstimator]:
        """Fit a list of models on the given data.

        Parameters
        ----------
        models : list of sklearn.base.BaseEstimator
            The models to fit.
        X : array-like
            The training data.
        y : array-like, optional
            The training labels.
        **fit_params : dict
            Additional parameters to pass to the model's fit method.

        Returns
        -------
        list of sklearn.base.BaseEstimator
            The fitted models.
        """
        pass

    def call_models(
        self,
        models: List[sklearn.base.BaseEstimator],
        method_name: str,
        *args,
        **kwargs,
    ) -> List:
        """Call a method on each model and return outputs."""
        outputs = []
        for model in models:
            method = getattr(model, method_name)
            outputs.append(method(*args, **kwargs))
        return outputs


@typeguard.typechecked
class DistributedComputationMixin(ABC):
    """Mixin class for backends that support distributed computation.

    This mixin provides the core logic for managing sweeps of models that can
    be trained in parallel across multiple processes or machines.
    """

    def __init__(
        self,
        trigger_type: str,
        num_workers: Optional[int],
        progress_bar: bool,
        sweep_type: str,
        sweep_name: Optional[str],
        cache_refresh_interval: float,
        fit_time_out: Optional[float],
        **kwargs,
    ):
        """Initialize the distributed computation mixin.

        Parameters
        ----------
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
            The type of sweep name to generate. 'uuid' generates a unique
            UUID for each sweep, while 'constant' uses the `sweep_name`.
        sweep_name : str, optional
            The name of the sweep to use when `sweep_type` is 'constant'.
            Required if `sweep_type` is 'constant'.
        cache_refresh_interval : float
            The interval in seconds at which to check the status of the sweep.
        fit_time_out : float, optional
            The maximum time in seconds to wait for the sweep to complete.
            If the timeout is reached, a `TimeoutError` is raised.
        """
        super().__init__()
        self.trigger_type = trigger_type
        self.num_workers = num_workers
        self.progress_bar = progress_bar
        self.sweep_type = sweep_type
        self.sweep_name = sweep_name
        self.cache_refresh_interval = cache_refresh_interval
        self.fit_time_out = fit_time_out
        self._validate_init_params()

    def _validate_init_params(self):
        """
        Validate initialization parameters.
        """
        if self.trigger_type == "auto" and self.num_workers is None:
            raise ValueError("num_workers must be specified when trigger_type is 'auto'")
        if self.sweep_type == "constant" and self.sweep_name is None:
            raise ValueError("sweep_name must be specified when sweep_type is 'constant'")

    def _generate_sweep_name(self):
        """
        Generate a unique UUID for a sweep.
        """
        return str(uuid.uuid4())

    def _get_sweep_name(self):
        """
        Get the sweep name.
        """
        if self.sweep_type == "uuid":
            return self._generate_sweep_name()
        elif self.sweep_type == "constant":
            return self.sweep_name
        else:
            raise ValueError(f"Invalid sweep type: {self.sweep_type}")

    def _trigger_sweep(self, sweep_name: str):
        """
        Trigger the sweep.
        """
        if self.trigger_type == "auto":
            self._run_sweep_multi_process(sweep_name)
        elif self.trigger_type == "manual":
            self._run_sweep_manual(sweep_name)
        else:
            raise ValueError(f"Invalid trigger type: {self.trigger_type}")

    @abstractmethod
    def _get_fit_distributed_cmd(self, sweep_name: str) -> List[str]:
        """Get the command to run the fit_distributed script.

        This method should be implemented by subclasses to provide the
        command-line instruction to start a worker process for a given sweep.

        Parameters
        ----------
        sweep_name : str
            The UUID of the sweep.

        Returns
        -------
        list of str
            The command to execute, with each element being a part of the
            command.
        """
        pass

    def _run_sweep_manual(self, sweep_name: str):
        """Trigger the sweep manually.

        This method should be implemented by subclasses to provide instructions
        for the user to start a worker process manually.

        Parameters
        ----------
        sweep_name : str
            The UUID of the sweep.
        """
        cmd = self._get_fit_distributed_cmd(sweep_name)

        command_str = " ".join(cmd)
        print("To start a worker process for the sweep manually, execute the following command:", flush=True)
        print(command_str + "\n", flush=True)

        return cmd

    def _run_sweep_multi_process(self, sweep_name: str):
        """
        Trigger the sweep by launching multiple non-blocking background worker processes.
        """
        # Base command for a single worker process.
        base_cmd = self._get_fit_distributed_cmd(sweep_name)

        # Store worker processes so we can wait for them before cleanup (avoids open file
        # handles on Windows when deleting temp dirs, e.g. in tests).
        self._current_worker_processes = []

        # Launch the command multiple times, once for each worker.
        for i in range(self.num_workers):
            # Find the next available counter for log files
            counter = 1
            while True:
                stdout_file = self._worker_log_dir / f"{sweep_name}_stdout_{counter}.log"
                stderr_file = self._worker_log_dir / f"{sweep_name}_stderr_{counter}.log"
                if not stdout_file.exists() and not stderr_file.exists():
                    break
                counter += 1

            with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                proc = subprocess.Popen(base_cmd, stdout=stdout, stderr=stderr)
                self._current_worker_processes.append(proc)

    @abstractmethod
    def _get_sweep_status(self, sweep_names: Union[str, List[str]], models_of_interest: Optional[List[str]] = None):
        """Get the status of a sweep.

        This method should be implemented by subclasses to return the status
        of one or more sweeps, including the number of completed, pending,
        and failed jobs.

        Parameters
        ----------
        sweep_names : str or list of str
            The UUID(s) of the sweep(s) to get the status for.
        models_of_interest : list of str, optional
            A list of model data hashes to get the status for. If not
            provided, the status for all models in the sweep is returned.

        Returns
        -------
        dict
            A dictionary containing the status of the sweep(s).
        """
        pass

    @abstractmethod
    def _retrieve_trained_model(self, model_data_hash: str) -> Optional[sklearn.base.BaseEstimator]:
        """Retrieve a trained model from the backend.

        Parameters
        ----------
        model_data_hash : str
            The hash of the model and data used for training.

        Returns
        -------
        sklearn.base.BaseEstimator or None
            The trained model, or None if the model is not found.
        """
        pass

    def _collect_trained_models(
        self,
        sweep_name: str,
        models_of_interest: List[str],
    ) -> List[Optional[sklearn.base.BaseEstimator]]:
        """Wait for sweep to finish and collect trained models."""

        status = self._get_sweep_status(sweep_name, models_of_interest)
        total = status["total"]
        start_time = time.time()

        # Select the real tqdm or the no-op version based on the flag
        progress_manager = tqdm(total=total, initial=status["completed"], desc="Fitting models") if self.progress_bar else utils.NoOpTqdm()

        with progress_manager as pbar:

            def update_progress_bar(status):
                pbar.update(status["completed"] - pbar.n)
                pbar.set_postfix(
                    {
                        "trained": f"{status['trained']}/{status['total']}",
                        "errors": status["errors"],
                        "reserved": status["reserved"],
                        "sweep_trained": f"{status['sweep_trained']}/{status['sweep_total']}",
                        "sweep_errors": f"{status['sweep_errors']}",
                        "sweep_reserved": f"{status['sweep_reserved']}",
                    }
                )

            update_progress_bar(status)

            while status["completed"] < total:
                time.sleep(self.cache_refresh_interval)
                status = self._get_sweep_status(sweep_name, models_of_interest)

                # Unconditionally update the progress bar and its details.
                update_progress_bar(status)

                if self.fit_time_out is not None and time.time() - start_time > self.fit_time_out:
                    raise TimeoutError(f"Fitting models timed out after {self.fit_time_out} seconds")

        # Collect models of interest
        trained_models = []
        for model_data_hash in models_of_interest:
            model = self._retrieve_trained_model(model_data_hash)
            trained_models.append(model)

        # Wait for worker processes to terminate so they release log file handles
        # (avoids PermissionError when deleting temp dirs on Windows).
        for proc in getattr(self, "_current_worker_processes", []):
            proc.wait()
        if hasattr(self, "_current_worker_processes"):
            self._current_worker_processes = []

        return trained_models
