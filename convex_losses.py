import math


# 0.5 * t^2
class SquaredSPPLoss:
    def solve_dual(self, alpha, beta):
        return beta / (1 + alpha)

    def eval(self, beta):
        return 0.5 * (beta ** 2)


# log(1+exp(t))
class LogisticSPPLoss:
    def solve_dual(self, alpha, beta):
        def qprime(s):
            return -alpha * s + beta + math.log(1 - s) - math.log(s)

        # compute [l,u] containing a point with zero qprime
        l = 0.5
        while qprime(l) <= 0:
            l /= 2.0

        u = 0.5
        while qprime(1 - u) >= 0:
            u /= 2.0
        u = 1 - u

        while u - l > 1E-10:  # should be accurate enough
            mid = (u + l) / 2
            if qprime(mid) == 0:
                return mid
            if qprime(l) * qprime(mid) > 0:
                l = mid
            else:
                u = mid

        return (u + l) / 2

    def eval(self, beta):
        return math.log(1 + math.exp(beta))
