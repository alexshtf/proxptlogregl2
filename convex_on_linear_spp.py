import torch


class ConvexOnLinearL2RegSPP:
    def __init__(self, x, eta, reg_coef, phi):
        self._eta = eta
        self._phi = phi
        self._reg_coef = reg_coef
        self._x = x

    def step(self, a, b):
        """
        Performs the optimizer's step, and returns the loss incurred.
        """
        eta = self._eta
        reg_coef = self._reg_coef
        phi = self._phi  # (*)
        x = self._x

        dot = torch.dot(a, x)
        val = phi.eval(dot.item()) + (reg_coef / 2) * (x.pow(2).sum().item())

        # compute the dual problem's coefficients
        alpha = eta * torch.sum(a ** 2) / (1 + eta * reg_coef)
        beta = dot / (1 + eta * reg_coef) + b

        # solve the dual problem
        s_star = phi.solve_dual(alpha.item(), beta.item())

        # update x
        x.sub_(eta * s_star * a)
        x.div_(1 + eta * reg_coef)

        # compute regularized loss0]
        return val

    def x(self):
        return self._x
