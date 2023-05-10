import time
import torch
x = torch.arange(-1, 1, 1e-7) # default cpu tensor
#measure CPU compute time by calling torch.erfinv
start_time = time.time()
torch.erfinv(x)
end = time.time()
cpu_time = end - start_time
print("CPU torch.erfinv time: ",cpu_time )

x = x.to("mps")
# measure MPS compute time
start_time = time.time()
torch.erfinv(x)
end = time.time()
mps_time = end - start_time
print("MPS torch.erfinv time: ", mps_time) 
print(f"MPS torch.erfinv is {cpu_time/mps_time*100} percent faster than CPU torch.erfinv")
