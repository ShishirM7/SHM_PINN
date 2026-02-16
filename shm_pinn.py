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
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, t):
        return self.net(t)

# 3. Training Preparation
model = PINN()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
t_physics = torch.linspace(0, 10, 2000).view(-1, 1).requires_grad_(True)

# 4. Training Loop
print("Starting Adam Optimization...")
for epoch in range(30000):
    optimizer_adam.zero_grad()
    
    # Same loss calculation as before
    t0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = model(t0)
    u0_t = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
    loss_bc = (u0 - 1.0)**2 + (u0_t - 0.0)**2
    
    u = model(t_physics)
    u_t = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_physics, torch.ones_like(u_t), create_graph=True)[0]
    residual = m*u_tt + mu*u_t + k*u
    loss_phs = torch.mean(residual**2)
    
    loss = loss_bc + 0.1*loss_phs # Increased BC weight
    loss.backward()
    optimizer_adam.step()
    
    if epoch % 1000 == 0:
        print(f"Adam Epoch {epoch} | Loss: {loss.item():.5f}")

# 2. Second Phase: L-BFGS (The "Scalpel")
print("\nSwitching to L-BFGS for fine-tuning...")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1, 
    max_iter=20000, 
    history_size=50,
    line_search_fn="strong_wolfe" # Standard trick for PINNs
)

def closure():
    optimizer_lbfgs.zero_grad()
    
    # Re-calculate IC loss
    u0_eval = model(t0)
    u0_t_eval = torch.autograd.grad(u0_eval, t0, torch.ones_like(u0_eval), create_graph=True)[0]
    loss_bc_eval = (u0_eval - 1.0)**2 + (u0_t_eval - 0.0)**2
    
    # Re-calculate Physics loss
    u_eval = model(t_physics)
    u_t_eval = torch.autograd.grad(u_eval, t_physics, torch.ones_like(u_eval), create_graph=True)[0]
    u_tt_eval = torch.autograd.grad(u_t_eval, t_physics, torch.ones_like(u_t_eval), create_graph=True)[0]
    residual_eval = m*u_tt_eval + mu*u_t_eval + k*u_eval
    loss_phs_eval = torch.mean(residual_eval**2)
    
    total_loss = 20*loss_bc_eval + loss_phs_eval
    total_loss.backward()
    return total_loss

optimizer_lbfgs.step(closure)
print("Final Loss after L-BFGS:", closure().item())

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