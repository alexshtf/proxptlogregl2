import torch


class ConvexOnLinearSPP:
    def __init__(self, x, eta, phi):
        self._eta = eta
        self._phi = phi
        self._x = x

    def step(self, a, b):
        """
        Performs the optimizer's step, and returns the loss incurred.
        """
        eta = self._eta
        phi = self._phi
        x = self._x

        # compute the dual problem's coefficients
        alpha = eta * torch.sum(a ** 2)
        beta = torch.dot(a, x) + b

        # solve the dual problem
        s_star = phi.solve_dual(alpha.item(), beta.item())

        # update x
        x.sub_(eta * s_star * a)

        return phi.eval(beta.item())

    def x(self):
        return self._x
