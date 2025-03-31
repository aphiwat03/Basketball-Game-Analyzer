import torch
print(torch.cuda.is_available())  # ตรวจสอบว่า PyTorch รองรับ CUDA หรือไม่
print(torch.cuda.get_device_name(0))  # ถ้ารองรับ CUDA จะบอกชื่อ GPU
