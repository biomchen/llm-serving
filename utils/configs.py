import os
import json

from .constants import (
    CONFIGS_PATH,
    LLM_CONFIGS
)

def load_configs(path):
    with open(path, 'r') as f:
        configs = json.load(f)
    f.close()
    return configs

class Paths:
    cwd = os.getcwd()
    CONFIGS_PATH = os.path.join(cwd, CONFIGS_PATH)

class ConfigsLoader(Paths):

    def __init__(self):
        super().__init__()
        self.TASK, self.MAX_LENGTH, self.TOP_K, self.TEMPERATURE, \
        self.TOP_P, self.DO_SAMPLE, self.NUM_RETURN_SEQUENCE \
            = self.load_llm_configs()
    
    def load_llm_configs(self):
        config_path = os.path.join(self.CONFIGS_PATH, LLM_CONFIGS)
        configs = load_configs(config_path)
        task = configs[0]['task']
        max_length = configs[0]['max_length']
        top_k = configs[0]['top_k']
        temperature = configs[0]['temperature']
        top_p = configs[0]['top_p']
        do_sample = configs[0]['do_sample']
        num_return_sequence = configs[0]['num_return_sequence']
        return task, max_length, top_k, temperature, top_p, do_sample, num_return_sequence

    
    