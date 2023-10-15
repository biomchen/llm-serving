import os
import json

def get_config_path():
    cwd = os.getcwd()
    return os.path.join(cwd, "config.json")

class ConfigLoader:

    def __init__(self):
        self.config_path = get_config_path()
        with open(self.config_path) as f:
            self.configs = json.loads(f.read())
    
    def load_version(self):
        return self.configs["version"]
        
    def list_models(self):
        return self.configs["models"]

    def load_llama2_7b_configs(self):
        llama2_7b_model = self.configs['llama2']['model_7b']['model']
        llama2_7b_cuda = self.configs['llama2']['model_7b']['cuda_device']
        llama2_7b_max_length = self.configs['llama2']['model_7b']['max_length']
        return llama2_7b_model, llama2_7b_cuda, llama2_7b_max_length

    def load_llama2_13b_configs(self):
        llama2_13b_model = self.configs['llama2']['model_13b']['model']
        llama2_13b_cuda = self.configs['llama2']['model_13b']['cuda_device']
        llama2_13b_max_length = self.configs['llama2']['model_13b']['max_length']
        return llama2_13b_model, llama2_13b_cuda, llama2_13b_max_length
    
    def load_codellama_7b_python_configs(self):
        codellama_7b_python_model = self.configs['llama2']['model_code_7b']['model']
        codellama_7b_python_cuda = self.configs['llama2']['model_code_7b']['cuda_device']
        codellama_7b_python_max_length = self.configs['llama2']['model_code_7b']['max_length']
        return codellama_7b_python_model, codellama_7b_python_cuda, \
            codellama_7b_python_max_length

    def load_codellama_13b_python_configs(self):
        codellama_13b_python_model = self.configs['llama2']['model_code_13b']['model']
        codellama_13b_python_cuda = self.configs['llama2']['model_code_13b']['cuda_device']
        codellama_13b_python_max_length = self.configs['llama2']['model_code_13b']['max_length']
        return codellama_13b_python_model, codellama_13b_python_cuda, \
            codellama_13b_python_max_length

    def load_llama_general_configs(self):
        llama2_task = self.configs['llama2']['task']
        llama2_do_sample = self.configs['llama2']['do_sample']
        llama2_top_k = self.configs['llama2']['top_k']
        llama2_num_return_sequence = self.configs['llama2']['num_return_sequence']
        return llama2_task, llama2_do_sample, llama2_top_k, llama2_num_return_sequence
    
class Configs(ConfigLoader):

    def __init__(self):
        super().__init__()
        self.VERSION = self.load_version()
        self.MODELS = self.list_models()
        self.LLAMA2_7B_MODEL, self.LLAMA2_7B_CUDA, \
            self.LLAMA2_7B_MAX_LEN = self.load_llama2_7b_configs()
        self.LLAMA2_13B_MODEL, self.LLAMA2_13B_CUDA, \
            self.LLAMA2_13B_MAX_LEN = self.load_llama2_13b_configs()
        self.CODELLAMA_7B_PYTHON_MODEL, self.CODELLAMA_7B_PYTHON_CUDA, \
            self.CODELLAMA_7B_PYTHON_MAX_LEN = self.load_codellama_7b_python_configs()
        self.CODELLAMA_13B_PYTHON_MODEL, self.CODELLAMA_13B_PYTHON_CUDA, \
            self.CODELLAMA_13B_PYTHON_MAX_LEN = self.load_codellama_13b_python_configs()
        self.LLAMA_TASK, self.LLAMA_DO_SAMPLE, self.LLAMA_TOP_K, \
        self.LLAMA_NUM_RETURN_SEQUENCE = self.load_llama_general_configs()

    
