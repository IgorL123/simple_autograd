import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import gradient as gr
import torch 

import gradient_ckeck
np.random.seed(42)

x2 = gr.Tensor([6])
x3 = gr.Tensor([0])
y = x2 * x2 + x2 * x3
y.backward()
grad_x2 = x2.grad
grad_x3 = x3.grad
grad_x2.backward()
grad_x2_x2 = x2.grad
grad_x2_x3 = x3.grad
x2_val = x2.numpy()
x3_val = x3.numpy()

print(y.numpy())
print(grad_x3.numpy())
print(grad_x2_x2.numpy())
print(grad_x2_x3)

print(grad_x2.inputs)
print(grad_x2_x2.inputs)
print(grad_x2_x3.inputs)