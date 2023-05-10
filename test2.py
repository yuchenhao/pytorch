import torch
x = torch.arange(-1, 1, 0.000001)
x = x.to("mps")
# measure MPS compute time
time = %timeit -o -q  torch.erfinv(x)
mps_time = time.average
print("MPS torch.erfinv time: ", mps_time)
x = torch.arange(-1, 1, 0.000001)
# measure CPU compute time by calling torch.erfinv but storing it to y_cpu
time = %timeit -o -q torch.erfinv(x)
cpu_time = time.average
print("CPU torch.erfinv time: ", cpu_time)
print(f"MPS torch.erfinv is {cpu_time/mps_time*100} percent faster than CPU torch.erfinv")

# compute MSE between y_cpu and y_mps
x = torch.arange(-1, 1, 0.000001)
x = x.to("mps")
y_mps = torch.erfinv(x)
y_cpu = torch.erfinv(x.to("cpu"))
mask = torch.isfinite(y_cpu) & torch.isfinite(y_mps.to("cpu"))
mse = torch.square(y_cpu[mask] - y_mps[mask].to("cpu")).mean()
print("MSE between MPS and CPU torch.erfinv: ", mse)