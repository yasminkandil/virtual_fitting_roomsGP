# import gc

# import torch
# def show_cache():
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     print(f"Memory allocated: {torch.cuda.memory_allocated()/(1024)} kb")
    
# show_cache()
# t=torch.randn(1024*256,requires_grad=True,device='cuda')    
# tc=t.cpu()
# dtc=t.detach().cpu()

# del t
# show_cache()

# del tc
# show_cache()
#  #%%   
# import gc
# def report_gpu():
#    print(torch.cuda.list_gpu_processes())
#    gc.collect()
#    torch.cuda.empty_cache()
# # %%
# import gc
# def report_gpu():
#    print(torch.cuda.list_gpu_processes())
#    gc.collect()
#    torch.cuda.empty_cache()
# # %%

import gc
from tkinter import Variable
from sklearn import model_selection


import torch
# gc.collect()
# print( torch.cuda.empty_cache())
# print(torch.cuda.is_available())
#print(torch.cuda.memory_summary(device='cuda', abbreviated=False))
def wipe_memory(self): # DOES WORK
    self._optimizer_to(torch.device('cpu'))
    del self.optimizer
    gc.collect()
    torch.cuda.empty_cache()

def _optimizer_to(self, device):
    for param in self.optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
