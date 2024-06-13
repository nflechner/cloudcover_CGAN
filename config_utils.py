import json
import os
from typing import Dict


def get_pipeline_config() -> Dict:
    '''
        Retrieve configuration for the current pipeline.

        Returns
        -------
        pipeline_config: Dict
            Dictionary with the pipeline config in JSON.
    '''
    config_path = os.path.join(os.getcwd(), 'pipeline_config.json')

    with open(config_path, 'r') as config_file:
        return json.load(config_file)