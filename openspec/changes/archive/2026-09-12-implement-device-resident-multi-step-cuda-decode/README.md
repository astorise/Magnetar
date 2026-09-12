# implement-device-resident-multi-step-cuda-decode

Make CUDA multi-step decode work by dispatching KV-history concatenation through a portable, device-resident Operator instead of downloading historical KV tensors to host, closing the last gap the Tachyon scope charter and prior audits identified before Tachyon-Mesh cutover.
