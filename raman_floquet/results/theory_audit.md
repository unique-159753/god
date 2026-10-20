# Independent convention and interpretation audit

This audit uses the Hamiltonian and Fourier convention in the pasted task.
The unavailable earlier Raman `fig2_corrected_Douter_comparison` data and
selection rules cannot be inferred from the earlier, different SVQL model.

## Operator and Fourier signs

Let `P[j,(j+1)%L]=1`, `L=89`, and order the basis site by site as up, down.
Then

```
T  = -(P + P†) tensor sigma_z
S  =  (P - P†) tensor (i sigma_y)
J2 = i(P² - P†²) tensor sigma_x.
```

All three operators are Hermitian. T and S are real symmetric; J2 is imaginary
Hermitian. With periodic boundaries, P and P† commute, giving exactly

```
[T,S] = 2 i J2.
```

For `H(tau)=sum_l H_l exp(+i*l*omega*tau)`, the first van Vleck correction is
`[H_+1,H_-1]/omega`, **in that order**. Therefore

```
[H_+1,H_-1]/omega = +A*B*sin(phi)/omega * J2.
```

An independent dense calculation found zero Hermiticity and `[T,S]-2iJ2`
residuals, and a Frobenius residual `9.24e-17` for the coefficient identity at
`phi=pi/3`. The opposite commutator ordering would reverse the physical sign.
The sign can also be calibrated using a rotating two-level field
`a*(sigma_x*cos(omega*tau)+sigma_y*sin(omega*tau))`, whose leading effective
Hamiltonian is `-a²*sigma_z/omega` in this convention.

The leading kick is

```
K1(tau) = A/omega*sin(omega*tau)*T
        + B/omega*sin(omega*tau+phi)*S.
K1(0)   = B/omega*sin(phi)*S.
```

The lab-frame effective eigenvector at drive phase zero is
`exp(-i*K1(0)) @ v_eff`. The sign follows from
`U(tau,0)=exp(-iK1(tau))*exp(-iH_eff*tau)*exp(+iK1(0))`.
Changing a numerical eigensolver's projection angle is not changing the
physical drive phase.

The compensation model `H_D(tau)=H(tau)-g_F*J2` cancels the leading van Vleck
term but not the leading kick. Its leading lab-frame reference is therefore
`exp(-i*K1(0))*eig(H0)`, rather than bare `eig(H0)`. In addition to the requested
static model A, retaining this kick-corrected static diagnostic prevents
attributing a residual micromotion contribution to higher-order resonances.

## Exact symmetries and one-photon limitations

Real static eigenvectors give

```
|M_mn(phi)|² = [A²*T_mn²+B²*S_mn²+2AB*T_mn*S_mn*cos(phi)]/4.
```

For any fixed frequency, eta, and phase-independent static-state mask,
both full-spectrum and static-outer resonance diagnostics are exactly
`C0(omega)+C1(omega)*cos(phi)`. They cannot have a strict interior phase
maximum unless the coefficient vanishes and the dependence is flat. Thus an
anomaly concentrated at an intermediate phase cannot be explained solely by
this summed one-photon matrix-element strength. This does not exclude a role
for one-photon resonances in a full dynamical mechanism.

The interaction-picture phase for an `H_l` matrix element is
`exp(i*(E_m-E_n+l*omega)*tau)`. Absorption with `E_m-E_n≈+omega` therefore uses
`H_-1`, as requested.

The two broadened diagnostics obey `R_norm = eta² * R` exactly at fixed eta;
they are different scalings of the same data, not independent tests.
`Tr(T²)/N=Tr(S²)/N=2`, `Tr(TS)/N=0`, so the full unmasked Fourier strength is
`Tr(H_-1† H_-1)/N=(A²+B²)/2=0.64`. Removing diagonal pairs or using the
static-outer mask only lowers it. Hence `0 <= R_norm <= 0.64` for this model.
R is a broadened response diagnostic, not an absorption rate or probability.

Because all instantaneous Hamiltonians are real,
`U(-phi)=U(phi)^T`, and the exact phase-zero Floquet IPRs are even in phi after
proper eigenstate matching. Also `C=I tensor sigma_x` anticommutes with every
instantaneous Hamiltonian, so `C U* C=U`. Exact quasienergies occur in ± pairs;
the partner vectors are `C v*` and have identical site and component IPRs.
For generic nonzero relative phase, `H(T-tau) != H(tau)`: the earlier SVQL
complex-symmetric-U/real-eigh trick is not applicable.

## Folding and outer-state selection

Independent static diagonalization gave

```
E_min = -3.298404128252782
E_max = +3.298404128252781
2*max|E| = 6.596808256505564.
```

The folding reference lies in the frequency interval under investigation.
When an energy crosses a Floquet-zone boundary, a rule that repeatedly picks
the highest `|epsilon|` can change the physical states included in an outer
average. A small exact/effective spectral difference can then create a large
hard-selection difference without a corresponding change in a tracked state.
Use the same centered zone `[-omega/2,+omega/2)` and the same rank/count rule
in all models; save selected masks, not just their mean D.

For 178 states, `ceil(0.2*N)=36` gives 18 particle-hole pairs and avoids
arbitrarily splitting an equal-absolute-energy pair. The task has explicitly
specified the static outer mask and its at-least-one-endpoint pair rule; the
earlier Floquet outer rule still needs documentation or an explicit working
definition.

An operational selection-bias decomposition is available. Match exact and
lab-frame effective modes by a maximum-overlap Hungarian assignment beta(alpha).
Then, for the exact outer mask A and effective outer mask B,

```
Delta_hard = mean_A D_exact - mean_B D_eff
Delta_matched = mean_A [D_exact(alpha)-D_eff(beta(alpha))]
Delta_selection = mean_A D_eff(beta(alpha)) - mean_B D_eff
Delta_hard = Delta_matched + Delta_selection.
```

Record overlaps and near-degenerate groups, since a low-overlap assignment is
not a reliable unique branch label. Frequency continuation using neighboring
mode overlaps gives a second check. A smooth alternative uses static-outer
projector weights `w_alpha=sum_{n in static_outer}|<n|v_alpha(0)>|²` and
`D_outer_projected=sum_alpha w_alpha*D_alpha/36`; completeness gives
`sum_alpha w_alpha=36`. This is a robustness diagnostic, not the same observable
as a hard Floquet-zone outer selection.

Spatial and component IPRs should be distinguished:

```
IPR_site = sum_j (p_up(j)+p_down(j))²
IPR_component = sum_j (p_up(j)²+p_down(j)²)
IPR_component <= IPR_site <= 2*IPR_component.
```

A component-IPR change can partly reflect spin redistribution rather than
spatial expansion. At one finite L, `-log(IPR)/log(L)` is a finite-size
participation exponent, not by itself a thermodynamic fractal dimension or
evidence of a new transition.

## Floquet-Sambe reconstruction and duplicate control

Expand `u(tau)=sum_m u_m exp(+i*m*omega*tau)`. The eigenproblem of
`H-i*d/dtau` is exactly the task's

```
HF[m,n] = H_(m-n) + m*omega*delta_mn.
```

Reconstruct the phase-zero physical vector as `u(0)=sum_m u_m` and normalize
that vector before calculating IPR or mode overlap. Save its norm before
normalization too: Sambe normalization fixes the period-averaged norm, while
finite photon truncation can make the reconstructed norm time dependent.

Gauge-equivalent copies obey
`epsilon' = epsilon + q*omega`, `u'_m = u_(m-q)` and have the same `u(0)`.
They must not be counted as new bands. Selecting the N largest central-sector
weights alone is unsafe near the folding reference: a correctly folded
outer static state can be dominated by photon sector +1 or -1.

Prefer Sambe eigenvalues already in the centered zone, reconstruct their
phase-zero vectors, and match them to exact Floquet modes using both
quasienergy distance and overlap. If the number of candidate eigenvalues is
not N, report it and resolve candidates against the N exact modes; never
silently interpret a surplus as additional physical bands. A rectangular
Hungarian assignment gives one candidate per exact mode, with overlap,
quasienergy error, edge-sector weight, and reconstructed norm recorded.
Near exact degeneracies compare projectors/subspaces, not arbitrary single
eigenvectors.

If a raw candidate outside the zone is used, first align its photon gauge.
For `q=(epsilon_raw-epsilon_fold)/omega`,
`u_fold,m = u_raw,m+q`. Photon-sector weights are gauge dependent and cannot
be compared across M values before this alignment. A truncation after gauge
shifting can discard weight, so report discarded/edge-sector weight.

Compare sectors ±1, ±2, and further sectors when needed against exact Floquet
data. M=1 is particularly vulnerable when the physical folded branch is
dominated by its edge sector. A visible avoided crossing or nonzero sideband
weight alone is not evidence of a new mechanism; quantify its contribution
to the same quasienergy-resolved localization observable.

The photon cutoff is denoted `m_max` in Figure 5, to distinguish it from the
number of time slices used for exact propagation. At `(omega,phi)=(6.2,0)` and
`(6.2,pi/3)`, cutoff 1 gives only 162 and 166 raw centered-zone candidates.
Cutoffs 2 and 3 give 178 at all four requested representative settings.
At cutoff 3, the outer-mean D difference from exact 128-slice propagation is
at most about `1.2e-5` and the quasienergy difference at most `1.5e-6`.
Reconstructed-wavefunction operator residuals are still `1e-5`–`1e-4`;
the latter must not be described as machine-precision convergence.

At omega=6.2, `2*omega=12.4` exceeds the complete static bandwidth `6.5968`.
Consequently there is no bare two-photon resonance between two static states
at energy separation `2*omega`. Higher photon-sector weights can diagnose
virtual processes or corrections to one-photon mixing; their presence alone
cannot establish a new two-photon transition.
