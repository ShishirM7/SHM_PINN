import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Physics Parameters
m = 1.0
mu = 0.5
k = 40.0

def exact_solution(t):
    # Analytical solution for an underdamped harmonic oscillator
    damp = -mu / (2*m)
    omega = np.sqrt(k/m - (mu/(2*m))**2)
    return np.exp(damp * t) * np.cos(omega * t)

# 2. Define the Neural Network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

# 3. Training Preparation
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
t_physics = torch.linspace(0, 10, 2000).view(-1, 1).requires_grad_(True)

# 4. Training Loop
for epoch in range(15000):
    optimizer.zero_grad()
    
    # Loss 1: Boundary/Initial Conditions (t=0, u=1, u'=0)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = model(t0)
    u0_t = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
    loss_bc = (u0 - 1.0)**2 + (u0_t - 0.0)**2
    
    # Loss 2: Physics Residual (m*u'' + mu*u' + k*u = 0)
    u = model(t_physics)
    u_t = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_physics, torch.ones_like(u_t), create_graph=True)[0]
    
    residual = m*u_tt + mu*u_t + k*u
    loss_phs = torch.mean(residual**2)
    
    # Total Loss
    loss = loss_bc + 0.1 * loss_phs # Weighting physics slightly less to help convergence
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

# 5. Visualization
t_test = np.linspace(0, 10, 2000)
u_exact = exact_solution(t_test)
with torch.no_grad():
    u_pinn = model(torch.tensor(t_test, dtype=torch.float32).view(-1, 1)).numpy()

plt.figure(figsize=(10, 5))
plt.plot(t_test, u_exact, label="Exact Solution", color='black', linestyle='--')
plt.plot(t_test, u_pinn, label="PINN Prediction", color='red', alpha=0.6)
plt.title("Damped Harmonic Oscillator: PINN vs Analytical")
plt.legend()
plt.show()