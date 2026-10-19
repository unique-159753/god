The calculation treats the single-particle Hilbert space of the driven Raman lattice exactly in space, with midpoint time discretization as its controlled numerical approximation. Parameters are L=89, alpha=55/89, t0=1, Delta0=0.5, V=2.5, W=0.01, A=B=0.8, theta=0, hbar=1, and periodic boundaries. The basis is ordered by site, with spin up preceding spin down. All reported real-space IPRs and dimensions are evaluated in this original 178-component basis.

The oriented nearest-neighbor matrix elements are T[j up,(j+1) up]=-1, T[j down,(j+1) down]=+1, S[j up,(j+1) down]=+1, and S[j down,(j+1) up]=-1, together with their Hermitian conjugates. The onsite term is [V cos(2 pi alpha j)+W] sigma_z, with physical sites j=1,...,L. Next-nearest-neighbor matrix elements of J2x are +i sigma_x on the oriented j to j+2 bond and its Hermitian conjugate on the reverse bond. Rational cosine arguments are reduced modulo L to preserve the exact reflection identity in floating-point arithmetic.

We use the Fourier convention
\[
H(\tau)=\sum_m H_m e^{+im\omega\tau},\qquad
H_{\pm1}=\frac12(AT+Be^{\pm i\varphi}S).
\]
Directly constructed physical-space matrices give
\[
[T,S]=2iJ_2^x,\qquad
[H_{+1},H_{-1}]=AB\sin\varphi\,J_2^x.
\]
The first-order van Vleck Hamiltonian is therefore H_vV=H0+(AB/omega) sin(phi) J2x. This sign corresponds to Delta(tau)=Delta0+B cos(omega tau+phi), including Delta(tau)=Delta0-B sin(omega tau) at phi=pi/2.

The leading kick is
\[
K^{(1)}(\tau)=\frac A\omega\sin(\omega\tau)T
+\frac B\omega\sin(\omega\tau+\varphi)S,
\qquad K_0=\frac{B\sin\varphi}{\omega}S.
\]
The lab-frame Floquet decomposition is U_F approximately exp(-i K0) exp(-i H_vV T) exp(+i K0), with T=2 pi/omega, so dressed eigenvectors are exp(-i K0) times effective eigenvectors. The kick exponential is evaluated by Hermitian diagonalization, without linearizing its exponential. Its sign and the effective-Hamiltonian sign are independently checked against full 178-dimensional, 256-slice propagators. On omega=16,32,64 at phi=pi/2, the normalized Frobenius propagator error scales approximately as omega^-2.024 without dressing and omega^-2.945 with correct dressing; reversing the kick gives omega^-1.964. These fits certify the convention, rather than imposing a uniform improvement on every localization observable.

For the reference calculation, at midpoint x_m=2 pi(m+1/2)/M we diagonalize the complete instantaneous Hamiltonian H_m and form
\[
U_F^{(M)}=e^{-iH_{M-1}T/M}\cdots e^{-iH_1T/M}e^{-iH_0T/M}.
\]
Later times multiply on the left. No effective Hamiltonian enters this reference evolution. The exact reflection-spin symmetry Q=R sigma_z at theta=0 is used to block-diagonalize every instantaneous matrix into two 89-dimensional sectors. R reflects the physical site index j to -j modulo L. This is an exact unitary change of basis, not a physical approximation. A complex Schur decomposition of each sector yields an orthonormal numerical eigenbasis; vectors are transformed back to the physical 178-component basis before evaluating localization. Symmetry sectors remain separate for all overlap assignments and repeated-eigensolve checks.

Quasienergies and observables are
\[
\varepsilon_n=-\arg\lambda_n/T,\quad q_n=\varepsilon_n/\omega\in[-1/2,1/2),\quad
\mathrm{IPR}_n=\sum_{j,\sigma}|\psi_{n,j\sigma}|^4,\quad
D_n=-\ln\mathrm{IPR}_n/\ln178.
\]
Effective energies are mapped by epsilon_eff=mod(E_eff+omega/2,omega)-omega/2. Stable sorting of absolute q selects ceil(0.2*178)=36 outer states; the central control selects the 36 smallest absolute q. Every parameter point uses its own model's quantile window. Selected counts, principal-zone ranges, and selection-coordinate ranges are saved. Ties follow stable eigenvalue ordering and are not assigned physical significance.

Four conventions are retained explicitly: (i) exact Floquet; (ii) effective bare states selected by largest absolute E_eff/omega without folding; (iii) identical effective bare states selected by largest absolute folded q; and (iv) dressed effective states using exactly the same folded indices as (iii). Thus (ii) versus (iii) changes only the selected states, whereas (iii) versus (iv) changes their lab-frame localization at fixed effective eigenstate labels. The difference curves are methodological comparisons, not independently additive physical correction terms.

The fresh initial scan uses 21 by 21 frequency-phase points over omega in [6,10] and phi in [0,pi], with M=64. The final 41 by 41 exact M=128 grid reuses the prior calculation with identical parameters; its original source and SHA256 are recorded in provenance.json, and three fresh pointwise recalculations reproduce its quasienergies and IPRs exactly in the present environment. All effective comparisons are newly computed. The bundled input_exact_grid128.npz makes the default workflow self-contained; run_study.py --fresh-exact-grid requests fresh evolution for the final grid instead of validated reuse. The original effective calculation already used folded quasienergies; reproducing its folded-bare grid differs by at most 2.22e-16. The unfolded-selection effective model is a new counterfactual comparison, not a repair of an earlier omission.

At phi=0,pi/4,pi/2, 401 fresh M=128 frequency points per phase cover [5.8,10]: spacing 0.005 on [5.8,6.6], 0.01 on [6.6,8], and 0.02 on [8,10]. Branches are anchored at omega=8 and tracked in both directions by Hungarian assignment maximizing total squared eigenvector overlap within each symmetry sector. Normalized eigenphases are lifted by q_lift,new=q_new+round(q_lift,previous-q_new); the displayed unwrapped quasienergy is omega*q_lift. Changing the anchor integer adds an integer times omega. The lift is a gauge convention, not a unique ordinary energy or photon number.

For the actual saved curves, q_lift-q is identically zero for every state at all sampled frequencies in [5.8,10]. The unwrapped and principal-zone energies therefore coincide numerically for this anchor; no exact branch-cut crossing is inferred merely from the static folding threshold. Zero integer lift does not exclude Floquet-state mixing or changes of eigenvectors. In tracked labels, the dynamic exact outer windows have zero membership turnover at phi=pi/4 and pi/2 throughout all 401 points. At phi=0 only one neighbor connection, ending at omega=6.2, exchanges two states. This exact-window turnover is distinct from the often much larger change between unfolded and folded effective-model windows.

Neighbor connections are flagged for assigned overlap below 0.8, assignment margin below 0.1, a same-sector circular q gap below 1e-6, or a change in eigenphase rank not explained by the common cyclic shift across the zone cut. Rank-swap detection is important because a narrow avoided crossing can be skipped even with high endpoint overlap. These flags do not establish a transition. Nine selected nonoverlapping suspect intervals receive midpoint-composition refinement through four subdivisions, adding 153 unique M=128 points. Assignments stabilize on the final two levels and agree with the coarse tracks, but two inspected phi=0 intervals, [6.080,6.085] and [6.200,6.205], remain connection-ambiguous. They are shaded in Figure 3 for any tracked state, even when the displayed representatives are smooth. No claim of adiabatic continuation is made. The fixed-reference outer-set evidence on [6.3,8] lies outside these intervals and contains no tracking flags for those reference states.

Mixing relative to the dressed effective basis is measured by P_n=[sum_m |<psi_eff,lab,m|psi_exact,n>|^4]^-1. This basis participation is distinct from the spatial IPR. Symmetry-respecting overlap matching also reports mean overlaps, matched-state D RMS errors, and window turnover, because small differences of window means can hide statewise discrepancies. The J2x control sets g_F to zero while keeping the same kick, and compares both dynamic windows and matched states. It isolates that term only inside the truncated effective model, not its exclusive contribution to the exact drive. Fixed absolute-q bins supplement quantile windows; empty bins are left blank.

Fresh validation evaluates M=32,64,128 at omega=6,6.3,6.5,7,8,10 and all three representative phases. It checks instantaneous Hermiticity, propagated unitarity, eigenvector normalization, and the actual eigen-equation residual against each returned U_F. Additional fresh M=128 points at omega=16,32,64 test high-frequency consistency. A sector-preserving permutation followed by an independent Schur solve tests basis reproducibility at the key M=128 points; arbitrary rotations inside exactly degenerate subspaces would still make individual-state IPR nonunique. The largest 64-to-128 outer-mean D change is 8.33e-5 in the key-point set. Higher-order kick and effective-Hamiltonian terms remain omitted from the comparison model, so correct first-order dressing need not decrease each D error: density changes and omitted terms can both begin at order omega^-2.

All figures use computed samples without interpolation. PNG output is 350 DPI. PDFs retain vector scatter, lines, text, heatmap cells, and colorbars; no image XObjects are present in the checked PDFs. Figure 1 uses phi=pi/2 and bare effective-state D for its effective spectrum colors. Figure 2 shares its D range across exact and corrected effective panels, and a symmetric zero-centered difference range across its two error panels. All numerical eigenstate, selection, branch, and validation data are exported as CSV and/or NPZ. Runtime records are stage-specific wall-clock measurements; concurrently executed stages must not be summed into a total elapsed time. The study establishes finite-size localization response, not a mobility edge or thermodynamic phase boundary.
