import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Unknown Parameter to Discover
k_actual = 40.0
m, mu = 1.0, 0.5

# 1. Generate Noisy Training Data (The "Observations")
t_obs = torch.linspace(0, 4, 25).view(-1, 1)
damp = -mu / (2*m)
omega = np.sqrt(k_actual/m - (mu/(2*m))**2)
u_true = np.exp(damp * t_obs.numpy()) * np.cos(omega * t_obs.numpy())
u_obs = torch.tensor(u_true, dtype=torch.float32) + 0.05*torch.randn(t_obs.shape)

# 2. Setup PINN with Learnable k
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, t): return self.net(t)

model = PINN()
k_learnable = nn.Parameter(torch.tensor([1.0], requires_grad=True)) # Initial guess: 1.0
optimizer = torch.optim.Adam(list(model.parameters()) + [k_learnable], lr=1e-3)
t_physics = torch.linspace(0, 10, 1000).view(-1, 1).requires_grad_(True)

for epoch in range(12001):
    optimizer.zero_grad()
    
    # Loss A: Data Loss (Match the noisy sensors)
    loss_data = torch.mean((model(t_obs) - u_obs)**2)
    
    # Loss B: Physics Loss (Force u to follow m*u'' + mu*u' + k_learnable*u = 0)
    u = model(t_physics)
    u_t = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_physics, torch.ones_like(u_t), create_graph=True)[0]
    residual = m*u_tt + mu*u_t + k_learnable*u
    loss_phs = torch.mean(residual**2)
    
    loss = 50*loss_data + loss_phs # Weight data loss higher to guide the k discovery
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Discovered k: {k_learnable.item():.3f}")

# Plotting Results
t_test = torch.linspace(0, 4, 200).view(-1, 1)
plt.scatter(t_obs, u_obs, label="Sensor Data", alpha=0.5)
plt.plot(t_test, model(t_test).detach().numpy(), color="green", label="PINN Fit")
plt.title(f"Inverse Problem: Discovered k ≈ {k_learnable.item():.2f}")
plt.legend(); plt.show()