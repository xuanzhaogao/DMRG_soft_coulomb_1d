# DMRG Benchmark for 1D Soft-Coulomb Atoms

## Method

We compute ground-state energies of 1D soft-Coulomb atoms (H through O) using the
Density Matrix Renormalization Group (DMRG) implemented with ITensor.jl.

**Hamiltonian.** Under the Born-Oppenheimer approximation with a single nucleus fixed
at the origin, the Hamiltonian reads

$$
H = \sum_{i=1}^{N}  [-\frac{1}{2}\nabla_{r_i}^{2} - z\,v(|r_i|)]  + \sum_{i \leq j} v(r_{ij}), \qquad v(u)=\frac{1}{\sqrt{1+u^{2}}}
$$

where $z$ is the nuclear charge, $N$ the electron count, and $v(u)$ is the soft-Coulomb
kernel.

**Discretization.** A uniform real-space grid on $[-r_c,\,r_c]$ with spacing
$\delta=0.1$ a.u. maps the continuous problem onto a lattice with $L=2r_c/\delta$
sites. Each site carries four states
($|\text{Emp}\rangle,|\!\uparrow\rangle,|\!\downarrow\rangle,|\!\uparrow\downarrow\rangle$)
with conserved particle number and $S_z$ quantum numbers. The kinetic energy is
discretized via the second-order finite-difference stencil.

**Long-range interaction via SOE-MPO.** Naively encoding the electron-electron
interaction produces an MPO with bond dimension $O(L)$, which is prohibitive for
$L=400$ sites. We instead decompose the soft-Coulomb kernel as a sum of exponentials
(SOE):

$$v(u)\approx\sum_{k=1}^{K}w_k\,e^{-s_k\,u}$$

with $K=35$ terms fitted by conditioned VARPRO, achieving a maximum absolute error of
$1.7\times10^{-7}$ on $u\in[0,\,40]$. Each exponential contributes a constant
increment to the MPO bond dimension, yielding an overall MPO bond dimension of
$D=6+K=41$, independent of $L$.

**DMRG protocol.** A ramp-up phase of 7 sweeps gradually increases the MPS bond
dimension through $\chi=50,100,200,\ldots$ with diminishing noise terms. This is
followed by batches of 5 sweeps at the target $\chi_{\max}=1000$ until the energy
change per batch falls below $10^{-8}$ Hartree.

## Results

Reference energies are taken from Wu et al. (arXiv:2603.23897). H and He references
originate from SG-CI calculations (Ref. [59] therein); Li–O references are
extrapolated from their SOG-TNN convergence study.

| System | $N$ | $z$ | $L$ | $E_{\text{DMRG}}$ (Hartree) | $E_{\text{ref}}$ (Hartree) | $\Delta E$ (mHartree) | $\chi_{\text{conv}}$ | Sweeps |
|--------|-----|-----|-----|-----------------------------|----------------------------|-----------------------|----------------------|--------|
| H      | 1   | 1   | 300 | -0.6698595509728            | -0.6697771382138           | -0.082                | 15                   | 15     |
| He     | 2   | 2   | 260 | -2.2385477942629            | -2.2382578241080           | -0.290                | —                    | 15     |
| Li     | 3   | 3   | 400 | -4.2112995470023            | -4.2105316476132           | -0.768                | —                    | 57+    |
| Be     | 4   | 4   | 400 | -6.7868011469427            | -6.7850779517085           | -1.723                | 67                   | 87     |
| B      | 5   | 5   | 400 | -9.8228972726154            | -9.8197124629442           | -3.185                | 138                  | 117    |
| C      | 6   | 6   | 400 | -13.3369332928517           | -13.3315431445955          | -5.390                | 364                  | 192    |
| N      | 7   | 7   | 400 | -17.3001785895317*          | -17.2916470936870          | -8.531*               | ~600*                | 132+*  |
| O      | 8   | 8   | 400 | -21.7114042086978*          | -21.6995705618073          | -11.834*              | 1000*                | 95+*   |

\* Still running; energy not yet converged.
