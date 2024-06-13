from os import listdir

# library imports
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
from pytorch_lightning.callbacks import ModelCheckpoint

# mlops imports
import mlflow
from mlflow.models import infer_signature
from lightning.pytorch.loggers import MLFlowLogger
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
from zenml.integrations.aws.orchestrators.sagemaker_orchestrator import SagemakerOrchestratorSettings
from zenml.client import Client

# class and function imports
from steps.data_prep import Data
from steps.CGAN import GAN_Model
from utils import log_utils, config_utils

# zenML imports
from zenml import step
from zenml.integrations.pytorch.materializers.pytorch_dataloader_materializer import PyTorchDataLoaderMaterializer
from zenml.integrations.pytorch_lightning.materializers.pytorch_lightning_materializer import PyTorchLightningMaterializer

"""
CODE
"""

# MLflow settings
mlflow_config = config_utils.get_pipeline_config().get('mlflow', {})
mlflow_settings = MLFlowExperimentTrackerSettings(experiment_name = mlflow_config.get('experiment_name'))
mlflow_tracker = mlflow_config.get('zenml_tracker_name')

# data settings (configuring s3 filepath)
data_config = config_utils.get_pipeline_config().get('data', {})
s3_path = data_config.get('s3_path')
sagemaker_settings = SagemakerOrchestratorSettings(
    input_data_s3_mode="File",  # TODO OR USE PIPE?
    input_data_s3_uri = s3_path,
    processor_args={
        "instance_type": "ml.p3.2xlarge",
        "volume_size_in_gb": 35
    }
)

@step(enable_cache = False,
        output_materializers = PyTorchDataLoaderMaterializer,
      settings={"orchestrator.sagemaker": sagemaker_settings})
def load_dataloaders() -> Tuple[
    Annotated[DataLoader, "train data loader"],
    Annotated[DataLoader, "validation data loader"],
    Annotated[DataLoader, "test data loader"]]:

    data_config = config_utils.get_pipeline_config().get('data', {})
    if data_config.get('current_mode') == 'local':
        dir_path = data_config.get('local_path')
    elif data_config.get('current_mode') == 's3':
        dir_path = "/opt/ml/processing/input/data"+"/" # unsure if the extra '/' is necessary (especially with pipe)
    else:
        log_utils.write_log(
            f"Data mode not recognized, proceeding with local training data at {data_config.get('local_path')}",
            'warning')
        dir_path = data_config.get('local_path')

    training_config = config_utils.get_pipeline_config().get('training', {})
    batch_size = training_config.get('batch_size')
    buffer = training_config.get('buffer')
    timeperiods = training_config.get('timeperiods')

    data = Data(dir_path, buffer, timeperiods)
    log_utils.write_log('Blocks were loaded and split into train, val and test sets')
    loader_train = DataLoader(data.train_data, shuffle=False, batch_size=batch_size, num_workers=3, persistent_workers=True) # iterates over [context_block]
    loader_val = DataLoader(data.val_data, shuffle=False, batch_size=batch_size, num_workers=3, persistent_workers=True) # iterates [context]
    loader_train = DataLoader(data.train_data, shuffle=False, batch_size=batch_size, num_workers=3, persistent_workers=True) # iterates [context]
    log_utils.write_log('Dataloaders were created')
    return loader_train, loader_val, loader_train

@step(enable_cache = False, 
    output_materializers = PyTorchLightningMaterializer, 
    settings={"orchestrator.sagemaker": sagemaker_settings})
def load_model() -> Annotated[pl.LightningModule, "GAN_model"]:
    model = GAN_Model()
    return model

@step(experiment_tracker=mlflow_tracker,
      settings = {'experiment_tracker.mlflow': mlflow_settings, "orchestrator.sagemaker": sagemaker_settings})
def training(model: pl.LightningModule, loader_train: DataLoader, loader_val: DataLoader, loader_test: DataLoader) -> None:

    run = mlflow.active_run()
    mlf_logger = MLFlowLogger(
        experiment_name=mlflow_config.get('experiment_name'), 
        run_id=run.info.run_id, 
        log_model="all", 
        tracking_uri=mlflow.get_tracking_uri()
        )

    training_config = config_utils.get_pipeline_config().get('training', {})
    epochs = training_config.get('epochs')

    trainer = pl.Trainer(logger=mlf_logger, max_epochs=epochs)
    mlflow.pytorch.autolog(log_datasets = False)
    trainer.fit(model, loader_train, loader_val)

    # test model
    test_config = config_utils.get_pipeline_config().get('testing', {})
    run_test = test_config.get('run_test')

    if run_test == "True":
        trainer.test(model, loader_test)

    
