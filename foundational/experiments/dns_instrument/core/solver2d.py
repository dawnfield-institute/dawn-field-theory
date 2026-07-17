"""
dns_instrument core -- 2D incompressible Navier-Stokes, vorticity form.

Pseudo-spectral on [0, 2pi)^2: dw/dt + u.grad(w) = nu * lap(w) (+ forcing).
Streamfunction psi_hat = w_hat / k2, u = (d psi/dy, -d psi/dx). Nonlinear
term computed in physical space with 2/3-rule dealiasing. RK4 in time.

Backend note: written against the module-level `xp = numpy`; the GPU port
replaces xp with torch (same call shapes) -- logic unchanged.
"""

import numpy as xp

TWO_THIRDS = 2.0 / 3.0


class Spectral2D:
    def __init__(self, n, nu):
        self.n = int(n)
        self.nu = float(nu)
        k1 = xp.fft.fftfreq(n, d=1.0 / n)          # integer wavenumbers
        self.kx = k1[:, None]
        self.ky = k1[None, :]
        self.k2 = self.kx**2 + self.ky**2
        self.k2_inv = xp.where(self.k2 > 0, 1.0 / xp.maximum(self.k2, 1e-300), 0.0)
        kmax = n // 2
        self.dealias = ((xp.abs(self.kx) < TWO_THIRDS * kmax) &
                        (xp.abs(self.ky) < TWO_THIRDS * kmax))

    def velocity(self, w_hat):
        psi_hat = w_hat * self.k2_inv
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        return xp.fft.ifft2(u_hat).real, xp.fft.ifft2(v_hat).real

    def rhs(self, w_hat):
        u, v = self.velocity(w_hat)
        wx = xp.fft.ifft2(1j * self.kx * w_hat).real
        wy = xp.fft.ifft2(1j * self.ky * w_hat).real
        adv_hat = xp.fft.fft2(u * wx + v * wy) * self.dealias
        return -adv_hat - self.nu * self.k2 * w_hat

    def step_rk4(self, w_hat, dt):
        k1 = self.rhs(w_hat)
        k2 = self.rhs(w_hat + 0.5 * dt * k1)
        k3 = self.rhs(w_hat + 0.5 * dt * k2)
        k4 = self.rhs(w_hat + dt * k3)
        return w_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- diagnostics (spectral, exact for band-limited fields) --

    def energy(self, w_hat):
        """E = 0.5 <|u|^2> via 0.5 * sum |w_hat|^2 / k^2 (Parseval)."""
        return 0.5 * float(xp.sum(xp.abs(w_hat)**2 * self.k2_inv)) / self.n**4

    def enstrophy(self, w_hat):
        """Z = 0.5 <w^2>."""
        return 0.5 * float(xp.sum(xp.abs(w_hat)**2)) / self.n**4

    def grid(self):
        x = xp.arange(self.n) * (2 * xp.pi / self.n)
        return xp.meshgrid(x, x, indexing='ij')


def taylor_green_w(X, Y, t, nu):
    """Exact 2D Taylor-Green vorticity: w = 2 sin x sin y exp(-2 nu t)."""
    return 2.0 * xp.sin(X) * xp.sin(Y) * xp.exp(-2.0 * nu * t)


def random_band_limited(n, kmin, kmax_band, seed):
    """Divergence-free-by-construction random vorticity, band-limited."""
    rng = xp.random.default_rng(seed)
    w_hat = xp.zeros((n, n), dtype=complex)
    phase = rng.uniform(0, 2 * xp.pi, (n, n))
    amp = rng.uniform(0.5, 1.0, (n, n))
    k1 = xp.fft.fftfreq(n, d=1.0 / n)
    kk = xp.sqrt(k1[:, None]**2 + k1[None, :]**2)
    band = (kk >= kmin) & (kk <= kmax_band)
    w_hat[band] = (amp * xp.exp(1j * phase))[band]
    w = xp.fft.ifft2(w_hat).real          # make it a real field
    w_hat = xp.fft.fft2(w)
    w_hat /= max(xp.sqrt(2 * 0.5 * xp.sum(xp.abs(w_hat)**2) / n**4), 1e-30)
    return w_hat
