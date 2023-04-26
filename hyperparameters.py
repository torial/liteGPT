import torch

"""
All models use a context window of 2048 tokens
Vocab size: 50257

MODEL                   n_params    n_layers    n_embed     n_heads     head_size       batch_size      learning_rate
GPT-3 Small             125M        12          768         12          64              0.5M            6.0e-4
GPT-3 Medium            350M        24          1024        16          64              0.5M            3.0e-4
GPT-3 Large             760M        24          1536        16          96              0.5M            2.5e-4
GPT-3 XL                1.3B        24          2048        24          128             1M              2.0e-4
GPT-3 2.7B              2.7B        32          2560        32          80              1M              1.6e-4
GPT-3 6.7B              6.7B        32          4096        32          128             2M              1.2e-4
GPT-3 13B               13.0B       40          5140        40          128             2M              1.0e-4
GPT-3 175B              175.0B      96          12288       96          128             3.2M            0.6e-4
"""

"""
Baseline NanoGPT:
# hyperparameters
batch_size = 64 # 
block_size = 256 # 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
# for 5000 iterations at learning rate of 1e-3:  (n_embed) (n_embed, time in seconds)
# (384, 230.6), (192, 187.5), (96, 196.4 seconds), (48, 186.6 seconds), (24, 199.6 seconds), (12, 195.5)
n_embed = 384 # 384 # 32
n_layer = 6 # 3
n_head = 6 # 4
dropout = 0.2
# --------------
"""


class HyperParameters:

    def __init__(self, name:str, batch_size:int, block_size:int, max_iters:int, eval_interval:int, learning_rate:float,
                 device: str, eval_iters:int, n_embed:int, n_layer:int, n_head:int, dropout:float):
        self.name = name
        self.batch_size = batch_size
        """how many independent sequences will we process in parallel?"""
        self.block_size = block_size
        """what is the maximum context length for predictions?"""
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.dropout = dropout


print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

hp_nano = HyperParameters("Nano", 64, 256, 5000, 500, 3e-4, device, 200, 384, 6, 6, 0.2)
hp_gpt3_small = HyperParameters("GPT-3 Small", 524_288, 2048, 5000, 500, 6e-4, device, 200, 768, 12, 12, 0.2)
hp_gpt3_medium = HyperParameters("GPT-3 Medium", 524_288, 2048, 5000, 500, 3e-4, device, 200, 1024, 24, 16, 0.2)
hp_gpt3_large = HyperParameters("GPT-3 Large", 524_288, 2048, 5000, 500, 2.5e-4, device, 200, 1536, 24, 16, 0.2)
hp_gpt3_xl = HyperParameters("GPT-3 XL", 1_048_576, 2048, 5000, 500, 2e-4, device, 200, 2048, 24, 24, 0.2)
hp_gpt3_27B = HyperParameters("GPT-3 2.7B", 1_048_576, 2048, 5000, 500, 1.6e-4, device, 200, 2560, 32, 32, 0.2)
hp_gpt3_67B = HyperParameters("GPT-3 6.7B", 2_097_152, 2048, 5000, 500, 1.2e-4, device, 200, 4096, 32, 32, 0.2)
hp_gpt3_130B = HyperParameters("GPT-3 13B", 2_097_152, 2048, 5000, 500, 1e-4, device, 200, 5140, 40, 40, 0.2)
hp_gpt3_1750B = HyperParameters("GPT-3 175B", 3_145_728, 2048, 5000, 500, 0.6e-4, device, 200, 12288, 96, 96, 0.2)
