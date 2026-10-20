"""Double-driven quasiperiodic Raman lattice, in the interleaved spin basis.

All propagation is time ordered, with full complex Floquet diagonalization.
The Fourier convention is H(tau)=sum_l H_l exp(+i*l*omega*tau).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import time
import ctypes
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import eigh, schur


@dataclass(frozen=True)
class Parameters:
    L: int = 89
    alpha: float = 55 / 89
    t0: float = 1.0
    delta0: float = 0.5
    V: float = 2.5
    W: float = 0.01
    A: float = 0.8
    B: float = 0.8
    theta: float = 0.0


Params = Parameters
DEFAULT_PARAMS = Parameters()


_NATIVE_SOURCE = r"""
#include <stddef.h>
#include <math.h>
/* Exact exp[-i h (t T_bond+delta S_bond)]; the two bond matrices commute. */
void bonds(double *u, size_t L, size_t cols, int group,
           double p, double q, double r, double s) {
    size_t start=group==2 ? L-1 : (size_t)group;
    size_t end=group==2 ? L : L-1;
    for(size_t site=start;site<end;site+=2) {
        size_t next=group==2 ? 0 : site+1;
        double *a=u+2*(2*site)*cols, *c=u+2*(2*site+1)*cols;
        double *b=u+2*(2*next)*cols, *d=u+2*(2*next+1)*cols;
        for(size_t j=0;j<2*cols;j+=2) {
            double ar=a[j],ai=a[j+1],cr=c[j],ci=c[j+1];
            double br=b[j],bi=b[j+1],dr=d[j],di=d[j+1];
            a[j]=p*ar-q*cr-r*bi+s*di;
            a[j+1]=p*ai-q*ci+r*br-s*dr;
            c[j]=p*cr-q*ar-s*bi+r*di;
            c[j+1]=p*ci-q*ai+s*br-r*dr;
            b[j]=p*br+q*dr-r*ai-s*ci;
            b[j+1]=p*bi+q*di+r*ar+s*cr;
            d[j]=p*dr+q*br+s*ai+r*ci;
            d[j+1]=p*di+q*bi-s*ar-r*cr;
        }
    }
}
void onsite(double *u,size_t L,size_t cols,const double *phase) {
    for(size_t site=0;site<L;site++) {
        double cr=phase[2*site],ci=phase[2*site+1];
        double *a=u+2*(2*site)*cols,*b=u+2*(2*site+1)*cols;
        for(size_t j=0;j<2*cols;j+=2) {
            double ar=a[j],ai=a[j+1],br=b[j],bi=b[j+1];
            a[j]=cr*ar-ci*ai; a[j+1]=cr*ai+ci*ar;
            b[j]=cr*br+ci*bi; b[j+1]=cr*bi-ci*br;
        }
    }
}
"""
_native_lock = threading.Lock()
_native_attempted = False
_native_library = None
_native_directory = None


def numerical_backend() -> str:
    """Initialize optional local C rotations; no downloads or fast-math."""
    global _native_attempted, _native_library, _native_directory
    with _native_lock:
        if _native_attempted:
            return "native_c" if _native_library is not None else "fft"
        _native_attempted = True
        if os.environ.get("RAMAN_BACKEND", "auto").lower() == "fft":
            return "fft"
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            return "fft"
        try:
            _native_directory = tempfile.TemporaryDirectory(prefix="raman_native_")
            source = os.path.join(_native_directory.name, "rotations.c")
            library = os.path.join(_native_directory.name, "rotations.so")
            with open(source, "w") as output:
                output.write(_NATIVE_SOURCE)
            result = subprocess.run([cc, "-O3", "-std=c99", "-fPIC", "-shared",
                                     "-ffp-contract=off", source, "-o", library],
                                    capture_output=True, timeout=30, check=False)
            if result.returncode:
                return "fft"
            lib = ctypes.CDLL(library)
            lib.bonds.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double]
            lib.bonds.restype = None
            lib.onsite.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_void_p]
            lib.onsite.restype = None
            _native_library = lib
        except (OSError, subprocess.SubprocessError):
            return "fft"
    return "native_c"


@lru_cache(maxsize=16)
def operators(params: Parameters = DEFAULT_PARAMS) -> dict[str, np.ndarray]:
    """Construct the exact PBC matrices; j starts at one, as in the request."""
    p = params
    if p.L < 3:
        raise ValueError("This nearest-neighbour PBC convention requires L>=3")
    shift = np.zeros((p.L, p.L))
    shift[np.arange(p.L), (np.arange(p.L) + 1) % p.L] = 1.0
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.diag([1., -1.])
    iy = np.array([[0., 1.], [-1., 0.]])
    T = -np.kron(shift + shift.T, sz)
    S = np.kron(shift - shift.T, iy)
    z = p.V * np.cos(2 * np.pi * p.alpha * np.arange(1, p.L + 1) + p.theta) + p.W
    HZ = np.kron(np.diag(z), sz)
    shift2 = shift @ shift
    J2 = 1j * np.kron(shift2 - shift2.T, sx)
    H0 = p.t0 * T + p.delta0 * S + HZ
    result = {"T": T, "S": S, "HZ": HZ, "J2": J2, "J2x": J2, "H0": H0,
              "shift": shift, "z": z}
    for value in result.values():
        value.setflags(write=False)
    return result


make_operators = operators


@lru_cache(maxsize=16)
def static_eigensystem(params: Parameters = DEFAULT_PARAMS) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = eigh(operators(params)["H0"], check_finite=False)
    values.setflags(write=False)
    vectors.setflags(write=False)
    return values, vectors


def fourier_components(phi: float, params: Parameters = DEFAULT_PARAMS):
    ops = operators(params)
    plus = 0.5 * (params.A * ops["T"] + params.B * np.exp(1j * phi) * ops["S"])
    return {0: ops["H0"], 1: plus, -1: plus.conj().T}


def g_f(omega: float, phi: float, params: Parameters = DEFAULT_PARAMS) -> float:
    return float(params.A * params.B * np.sin(phi) / omega)


def hamiltonian(tau: float, omega: float, phi: float,
                params: Parameters = DEFAULT_PARAMS, compensate: bool = False,
                j2_coefficient: float = 0.0) -> np.ndarray:
    ops = operators(params)
    coefficient = j2_coefficient - (g_f(omega, phi, params) if compensate else 0.0)
    return ((params.t0 + params.A * np.cos(omega * tau)) * ops["T"]
            + (params.delta0 + params.B * np.cos(omega * tau + phi)) * ops["S"]
            + ops["HZ"] + coefficient * ops["J2"])


def fold_quasienergies(values: np.ndarray, omega: float) -> np.ndarray:
    return (np.asarray(values) + omega / 2) % omega - omega / 2


def localization_observables(modes: np.ndarray) -> dict[str, np.ndarray]:
    """Return both real-space and component IPR conventions explicitly."""
    if modes.ndim != 2 or modes.shape[0] % 2:
        raise ValueError("Expected columns of interleaved-spin mode vectors")
    L = modes.shape[0] // 2
    probabilities = np.abs(modes.reshape(L, 2, modes.shape[1])) ** 2
    densities = probabilities.sum(axis=1)
    ipr_site = np.sum(densities ** 2, axis=0)
    ipr_component = np.sum(probabilities ** 2, axis=(0, 1))
    return {"ipr_site": ipr_site, "ipr_component": ipr_component,
            "D_site": -np.log(ipr_site) / np.log(L),
            "D_component": -np.log(ipr_component) / np.log(2 * L),
            "D_component_logL": -np.log(ipr_component) / np.log(L),
            "density": densities}


@lru_cache(maxsize=16)
def _wave_numbers(L: int):
    k = 2 * np.pi * np.fft.fftfreq(L)
    return np.cos(k), np.sin(k), np.sin(2 * k)


def _kinetic_rotation(u: np.ndarray, duration: float, t_value: float,
                      delta_value: float, j2_coefficient: float = 0.) -> np.ndarray:
    """Apply exp(-i*duration*Hkin) exactly in momentum spin blocks."""
    cosine, sine, sine2 = _wave_numbers(u.shape[0])
    hx = -2 * j2_coefficient * sine2
    hy = -2 * delta_value * sine
    hz = -2 * t_value * cosine
    radius = np.sqrt(hx * hx + hy * hy + hz * hz)
    c = np.cos(duration * radius)
    sr = duration * np.sinc(duration * radius / np.pi)
    aa = c - 1j * sr * hz
    bb = c + 1j * sr * hz
    ab = -1j * sr * (hx - 1j * hy)
    ba = -1j * sr * (hx + 1j * hy)
    # Keep the transform axis contiguous after the first step: each column
    # becomes a batch of two length-L spin wavefunctions. This matters for
    # prime L=89 and avoids repeated strided FFTs through a dense U matrix.
    momentum = fft(u.transpose(2, 1, 0), axis=2, workers=1)
    a, b = momentum[:, 0, :].copy(), momentum[:, 1, :].copy()
    momentum[:, 0, :] = aa[None, :] * a + ab[None, :] * b
    momentum[:, 1, :] = ba[None, :] * a + bb[None, :] * b
    return ifft(momentum, axis=2, workers=1).transpose(2, 1, 0)


def kick_operator(omega: float, phi: float, params: Parameters = DEFAULT_PARAMS,
                  tau: float = 0.0) -> np.ndarray:
    ops = operators(params)
    return (params.A / omega * np.sin(omega * tau) * ops["T"]
            + params.B / omega * np.sin(omega * tau + phi) * ops["S"])


def apply_kick(modes: np.ndarray, omega: float, phi: float,
               params: Parameters = DEFAULT_PARAMS, tau: float = 0.0) -> np.ndarray:
    """Apply the first-order lab-frame e^{-i K1(tau)} exactly, without series truncation."""
    shape = modes.shape
    rotated = _kinetic_rotation(modes.reshape(params.L, 2, -1), 1.,
                               params.A / omega * np.sin(omega * tau),
                               params.B / omega * np.sin(omega * tau + phi))
    return rotated.reshape(shape)


def _build_fft_operator(omega: float, phi: float, steps: int = 128,
                        params: Parameters = DEFAULT_PARAMS, compensate: bool = False,
                        order: int = 4, j2_coefficient: float = 0.0) -> np.ndarray:
    """Full U(T,0), by symmetric second- or fourth-order time-dependent splitting.

    A kinetic step is exact in k-space; HZ rotations are exact in real space.
    Yoshida composition advances the clock through negative substeps too.
    No time-reflection assumption is made for a general relative phase phi.
    """
    if not np.isfinite(omega) or omega <= 0 or int(steps) != steps or steps < 1:
        raise ValueError("Require omega>0 and positive integer steps")
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    period = 2 * np.pi / omega
    dt = period / steps
    z = operators(params)["z"]
    coefficient = j2_coefficient - (g_f(omega, phi, params) if compensate else 0.0)
    if order == 2:
        weights = (1.,)
    else:
        w1 = 1 / (2 - 2 ** (1 / 3))
        weights = (w1, 1 - 2 * w1, w1)
    phases = [np.exp(-.5j * dt * weight * z)[:, None] for weight in weights]
    u = np.eye(2 * params.L, dtype=np.complex128).reshape(params.L, 2, -1)
    for step in range(int(steps)):
        start = step * dt
        for weight, phase in zip(weights, phases):
            duration = weight * dt
            midpoint = start + duration / 2
            u[:, 0, :] *= phase
            u[:, 1, :] *= phase.conj()
            u = _kinetic_rotation(u, duration,
                                  params.t0 + params.A * np.cos(omega * midpoint),
                                  params.delta0 + params.B * np.cos(omega * midpoint + phi),
                                  coefficient)
            u[:, 0, :] *= phase
            u[:, 1, :] *= phase.conj()
            start += duration
    return u.reshape(2 * params.L, 2 * params.L)


def _build_native_operator(omega: float, phi: float, steps: int,
                           params: Parameters, order: int) -> np.ndarray:
    """Odd-ring even/odd/closing-bond splitting, with exact 4x4 bond gates."""
    if not np.isfinite(omega) or omega <= 0 or int(steps) != steps or steps < 1:
        raise ValueError("Require omega>0 and positive integer steps")
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    dt = 2 * np.pi / (omega * steps)
    if order == 2:
        weights = (1.,)
    else:
        w1 = 1 / (2 - 2 ** (1 / 3))
        weights = (w1, 1 - 2 * w1, w1)
    phases = [np.ascontiguousarray(np.exp(-1j * dt * w * operators(params)["z"]))
              for w in weights]
    u = np.eye(2 * params.L, dtype=np.complex128)
    pointer, L, cols = u.ctypes.data, params.L, u.shape[1]
    for step in range(int(steps)):
        start = step * dt
        for weight, phase in zip(weights, phases):
            duration = weight * dt
            midpoint = start + duration / 2
            tv = params.t0 + params.A * np.cos(omega * midpoint)
            dv = params.delta0 + params.B * np.cos(omega * midpoint + phi)
            ct, st = np.cos(tv * duration / 2), np.sin(tv * duration / 2)
            cd, sd = np.cos(dv * duration / 2), np.sin(dv * duration / 2)
            coefficients = (float(ct * cd), float(st * sd), float(st * cd), float(ct * sd))
            for group in (0, 1, 2):
                _native_library.bonds(pointer, L, cols, group, *coefficients)
            _native_library.onsite(pointer, L, cols, phase.ctypes.data)
            for group in (2, 1, 0):
                _native_library.bonds(pointer, L, cols, group, *coefficients)
            start += duration
    return u


def build_floquet_operator(omega: float, phi: float, steps: int = 128,
                           params: Parameters = DEFAULT_PARAMS, compensate: bool = False,
                           order: int = 4, j2_coefficient: float = 0.0,
                           backend: str = "auto") -> np.ndarray:
    """Construct complete U; C-bond or FFT splitting, both time ordered.

    ``backend='fft'`` is an independent spatial factorization. Any nonzero
    NNN compensation uses that general FFT method. The two backends differ
    at finite step count and converge to the same exact evolution.
    """
    coefficient = j2_coefficient - (g_f(omega, phi, params) if compensate else 0.0)
    if backend not in ("auto", "fft", "native", "native_c"):
        raise ValueError("backend must be auto, fft, native, or native_c")
    native = backend != "fft" and coefficient == 0. and numerical_backend() == "native_c"
    if native:
        return _build_native_operator(omega, phi, steps, params, order)
    return _build_fft_operator(omega, phi, steps, params, compensate, order, j2_coefficient)


def floquet_modes(u: np.ndarray, omega: float, diagnostics: bool = True,
                  return_modes: bool = True) -> dict[str, Any]:
    """Orthogonal Floquet modes from complex Schur decomposition of unitary U."""
    triangular, vectors = schur(u, output="complex", check_finite=False)
    eigenvalues = np.diag(triangular).copy()
    energies = fold_quasienergies(-np.angle(eigenvalues) * omega / (2 * np.pi), omega)
    order = np.argsort(energies)
    eigenvalues, energies, vectors = eigenvalues[order], energies[order], vectors[:, order]
    statistics = localization_observables(vectors)
    result: dict[str, Any] = {"quasienergies": energies,
                              "eigenvalue_modulus_error": float(np.max(abs(abs(eigenvalues) - 1))),
                              "schur_offdiagonal_error": float(np.linalg.norm(triangular - np.diag(np.diag(triangular))) / np.sqrt(u.shape[0])),
                              **statistics}
    phases = np.sort(np.mod(np.angle(eigenvalues), 2 * np.pi))
    gaps = np.diff(np.r_[phases, phases[0] + 2 * np.pi])
    result["min_eigenphase_gap"] = float(np.min(gaps))
    result["degenerate_eigenphase_gaps"] = int(np.count_nonzero(gaps < 1e-9))
    if diagnostics:
        result["unitarity_error"] = float(np.linalg.norm(u.conj().T @ u - np.eye(u.shape[0])) / np.sqrt(u.shape[0]))
        result["eigen_residual"] = float(np.linalg.norm(u @ vectors - vectors * eigenvalues[None, :]) / np.sqrt(u.shape[0]))
        result["orthogonality_error"] = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(u.shape[0])) / np.sqrt(u.shape[0]))
    if return_modes:
        result["modes"] = vectors
    return result


def exact_modes(omega: float, phi: float, steps: int = 128,
                params: Parameters = DEFAULT_PARAMS, compensate: bool = False,
                order: int = 4, diagnostics: bool = True,
                return_modes: bool = True, return_operator: bool = False,
                backend: str = "auto") -> dict[str, Any]:
    start = time.perf_counter()
    u = build_floquet_operator(omega, phi, steps, params, compensate, order, backend=backend)
    propagated = time.perf_counter()
    result = floquet_modes(u, omega, diagnostics, return_modes)
    result.update(omega=float(omega), phi=float(phi), steps=int(steps),
                  propagation_seconds=propagated - start,
                  eigensolve_seconds=time.perf_counter() - propagated,
                  total_seconds=time.perf_counter() - start,
                  compensate=bool(compensate),
                  backend=("native_c" if backend != "fft" and
                           (not compensate or g_f(omega, phi, params) == 0.) and
                           numerical_backend() == "native_c" else "fft"))
    if return_operator:
        result["operator"] = u
    return result


def effective_modes(omega: float, phi: float, params: Parameters = DEFAULT_PARAMS,
                    kick_corrected: bool = True, return_modes: bool = True,
                    include_first_order: bool = True) -> dict[str, Any]:
    """First-order van Vleck modes, folded before any outer-zone selection."""
    ops = operators(params)
    h_eff = ops["H0"] + (g_f(omega, phi, params) * ops["J2"] if include_first_order else 0.)
    energies, vectors = eigh(h_eff, check_finite=False)
    folded = fold_quasienergies(energies, omega)
    order = np.argsort(folded)
    energies, folded, vectors = energies[order], folded[order], vectors[:, order]
    lab_vectors = apply_kick(vectors, omega, phi, params) if kick_corrected else vectors
    result: dict[str, Any] = {"energies": energies, "quasienergies": folded,
                              "g_F": g_f(omega, phi, params),
                              "kick_corrected": bool(kick_corrected),
                              **localization_observables(lab_vectors)}
    if return_modes:
        result.update(modes=lab_vectors, effective_frame_modes=vectors)
    return result
