from zenml import pipeline
from steps.run_model import load_dataloaders, load_model, training
from zenml.config import DockerSettings
from utils import log_utils, config_utils

# Retrieve ZenML configuration
zenml_config = config_utils.get_pipeline_config().get('zenml', {})

# Retrieve requirements for installation in Docker image, but do not include
# ZenML itself, which comes from the configuration file.
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [x for x in requirements if 'zenml' not in x]
# requirements += [zenml_config.get("version")]

# Specify the Docker settings
docker_settings = DockerSettings(
    parent_image="..",
    requirements=requirements,
    build_options={
        "network": "host"
    }
)

@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def CGAN_2d_cs_20231030():
    loader_train, loader_val, loader_test = load_dataloaders()
    model = load_model()
    training(model, loader_train, loader_val, loader_test)

if __name__ == "__main__":
    CGAN_2d_cs_20231030()