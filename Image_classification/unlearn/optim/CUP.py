import torch
import numpy as np
    
class Cup(torch.optim.Optimizer):
    def __init__(self, optimizer, gamma):
        defaults = dict()
        super().__init__(optimizer.param_groups, defaults)
        self.optimizer = optimizer
        self.epsilon = 1e-6
        self.iter = 0
        self.gamma = gamma

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad   

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, losses):
        self.iter += 1
        with torch.enable_grad():
            forget_loss = losses[0]
        self.zero_grad()
        forget_loss.backward(retain_graph=True)
        forget_grad, forget_shape, forget_has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    forget_shape.append(p.shape)
                    forget_grad.append(torch.zeros_like(p).to(p.device))
                    forget_has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                forget_shape.append(p.grad.shape)
                forget_grad.append(p.grad.clone())
                forget_has_grad.append(torch.ones_like(p).to(p.device))
        flatten_forget_grad = self._flatten_grad(forget_grad)                

        with torch.enable_grad():
            retain_loss = losses[1]
        self.zero_grad()
        retain_loss.backward(retain_graph=True)
        retain_grad = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    retain_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                retain_grad.append(p.grad.clone())
        flatten_retain_grad = self._flatten_grad(retain_grad)    

        flatten_total_grad = (flatten_retain_grad + flatten_forget_grad)/2

        retain_norm_sq = flatten_retain_grad.dot(flatten_retain_grad).item() + self.epsilon
        forget_norm = torch.norm(flatten_forget_grad, p=2) + self.epsilon
        forget_norm_sq = forget_norm * forget_norm
        total_norm = torch.norm(flatten_total_grad, p=2)

        retain_forget_dot = flatten_retain_grad.dot(flatten_forget_grad).item()

        retain_proj = (flatten_retain_grad-(retain_forget_dot/forget_norm_sq)*flatten_forget_grad)
        forget_proj = (flatten_forget_grad-(retain_forget_dot/retain_norm_sq)*flatten_retain_grad)

        retain_proj /= (torch.norm(retain_proj, p=2)+self.epsilon)
        forget_proj /= (torch.norm(forget_proj, p=2)+self.epsilon)

        theta = torch.acos(torch.clamp(retain_proj.dot(forget_proj), -1, 1))

        gamma = self.gamma * theta

        g = torch.sin(gamma) * flatten_forget_grad/(forget_norm) + torch.cos(gamma) * retain_proj
        g = (total_norm)*g

        unflatten_update_grad = self._unflatten_grad(g, forget_shape)
        with torch.enable_grad():
            self.zero_grad()
            idx = 0
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param.grad = unflatten_update_grad[idx]
                    idx += 1

        self.optimizer.step()
        return