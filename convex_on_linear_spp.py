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

        # compute the dual problem's coefficients
        alpha = eta * torch.sum(a ** 2) / (1 + eta * reg_coef)
        beta = torch.dot(a, x) / (1 + eta * reg_coef) + b

        # solve the dual problem
        s_star = phi.solve_dual(alpha.item(), beta.item())

        # update x
        x.sub_(eta * s_star * a)

        # compute regularized loss0]
        return phi.eval(torch.dot(a, x).item()) + (reg_coef / 2) * (x.pow(2).sum().item())

    def x(self):
        return self._x
