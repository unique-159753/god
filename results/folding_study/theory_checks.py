"""Independent Fourier-sign, kick-sign, and static folding checks.

Conventions: H(t)=sum_m H_m exp(+i*m*omega*t),
U_F=exp(-i K(0)) exp(-i H_vV T) exp(+i K(0)) + O(omega^-3).
This standalone check constructs physical matrices without importing the main
calculation engine. It uses full Hilbert-space propagation, not a high-frequency
approximation, for its reference U_F.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh


def operators(L=89, p=55):
    n = 2 * L
    T = np.zeros((n, n), complex)
    S = np.zeros_like(T)
    J = np.zeros_like(T)
    def add(matrix, i, j, value):
        matrix[i, j] += value
        matrix[j, i] += np.conjugate(value)
    for j in range(L):
        k, r = (j + 1) % L, (j + 2) % L
        add(T, 2*j, 2*k, -1)
        add(T, 2*j+1, 2*k+1, 1)
        add(S, 2*j, 2*k+1, 1)
        add(S, 2*j+1, 2*k, -1)
        add(J, 2*j, 2*r+1, 1j)
        add(J, 2*j+1, 2*r, 1j)
    residue = (p * np.arange(1, L+1)) % L
    z = 2.5*np.cos(2*np.pi*np.minimum(residue, L-residue)/L)+.01
    H0 = T + .5*S + np.diag(np.repeat(z, 2)*np.tile([1, -1], L))
    return T, S, J, H0


def exp_minus_i_hermitian(H, duration=1.):
    E, V = eigh(H, check_finite=False, driver='evd')
    return (V*np.exp(-1j*E*duration))@V.conj().T


def exact_U(T, S, H0, omega, phi, slices=256):
    period = 2*np.pi/omega
    U = np.eye(T.shape[0], dtype=complex)
    # Instantaneous H is real; real eigensolvers reduce the independent cost.
    for m in range(slices):
        x = 2*np.pi*(m+.5)/slices
        H = (H0+.8*np.cos(x)*T+.8*np.cos(x+phi)*S).real
        E, V = eigh(H, check_finite=False, driver='evd')
        U = (V*np.exp(-1j*E*period/slices))@(V.T@U)
    return U


def run():
    started = time.perf_counter()
    T, S, J, H0 = operators()
    phi = np.pi/2
    Hp = (.8*T+.8*np.exp(1j*phi)*S)/2
    Hm = Hp.conj().T
    expected = .64*np.sin(phi)*J
    comm = Hp@Hm-Hm@Hp
    out = {
        'convention': 'H(t)=sum H_m exp(+i m omega t)',
        'commutator_T_S_minus_2iJ_max': float(np.max(abs(T@S-S@T-2j*J))),
        'fourier_commutator_minus_ABsinphiJ_max': float(np.max(abs(comm-expected))),
        'fourier_commutator_plus_ABsinphiJ_max': float(np.max(abs(comm+expected))),
        'hermiticity_max': float(max(np.max(abs(h-h.conj().T)) for h in (T,S,J,H0))),
        'physical_dressing': 'psi_lab=exp(-i K0) psi_eff; K0=B sin(phi) S / omega',
        'kick_sign_test': [],
    }
    E = eigh(H0, eigvals_only=True, check_finite=False)
    out['static_energy_min'] = float(E[0])
    out['static_energy_max'] = float(E[-1])
    out['static_max_abs_energy'] = float(max(abs(E)))
    out['static_fold_threshold'] = float(2*max(abs(E)))
    outer_n = int(np.ceil(.2*len(E)))
    bare = set(np.argsort(abs(E), kind='stable')[-outer_n:].tolist())
    out['static_selection'] = []
    for omega in (5.8, 6., 6.3, 6.5, 6.6, 7., 8., 10.):
        q = (E/omega+.5)%1-.5
        folded = set(np.argsort(abs(q), kind='stable')[-outer_n:].tolist())
        out['static_selection'].append({
            'omega': omega, 'outside_principal_zone_count': int(np.sum(abs(E)>omega/2)),
            'outer_count': outer_n, 'outer_replaced_by_folding': len(bare-folded),
        })
    for omega in (8., 16., 32., 64.):
        U = exact_U(T, S, H0, omega, phi, slices=256)
        Heff = H0 + expected/omega
        Ueff = exp_minus_i_hermitian(Heff, 2*np.pi/omega)
        K0 = .8*np.sin(phi)*S/omega
        minus = exp_minus_i_hermitian(K0)
        plus = minus.conj().T
        err = lambda v: float(np.linalg.norm(U-v, 'fro')/np.sqrt(len(U)))
        out['kick_sign_test'].append({
            'omega': omega, 'phi': phi, 'slices': 256,
            'unitarity_relative_fro': float(np.linalg.norm(U.conj().T@U-np.eye(len(U)), 'fro')/np.sqrt(len(U))),
            'bare_U_error': err(Ueff),
            'correct_minus_K_U_error': err(minus@Ueff@plus),
            'wrong_plus_K_U_error': err(plus@Ueff@minus),
            'wrong_Heff_correction_with_correct_K_U_error': err(minus@exp_minus_i_hermitian(H0-expected/omega, 2*np.pi/omega)@plus),
        })
    # Scaling of propagator error, not eigenstate IPR, certifies the expansion.
    high = out['kick_sign_test'][-3:]
    x = np.log([r['omega'] for r in high])
    out['high_frequency_error_power_laws'] = {}
    for key in ('bare_U_error', 'correct_minus_K_U_error', 'wrong_plus_K_U_error',
                'wrong_Heff_correction_with_correct_K_U_error'):
        y = np.log([r[key] for r in high])
        out['high_frequency_error_power_laws'][key] = float(-np.polyfit(x, y, 1)[0])
    out['seconds'] = time.perf_counter()-started
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run()
