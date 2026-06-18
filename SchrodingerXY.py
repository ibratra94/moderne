# -*- coding: utf-8 -*-
# =============================================================================
#  Projet "Effet tunnel" - Physique moderne
#  Programme 3 (partie B) : RESOLUTION NUMERIQUE de l'equation de SCHRODINGER
#                           dependante du temps (1D), schema explicite.
#
#  Equation de Schrodinger 1D dependante du temps :
#       i hbar dPsi/dt = -(hbar^2 / 2m) d2Psi/dx2 + V(x) Psi
#  Reecrite pour le pas de temps :
#       dPsi/dt = (i hbar / 2m) d2Psi/dx2 - (i / hbar) V(x) Psi
# -----------------------------------------------------------------------------
#  Schema FTCS (Forward Time, Centered Space) :
#  (Euler avant en temps + difference centree a 3 points en espace)
#    psi[n+1][j] = psi[n][j]
#                + 1j*hbar*dt/(2*m*dx**2) * (psi[n][j+1] - 2*psi[n][j] + psi[n][j-1])
#                - 1j*dt/hbar * V[j] * psi[n][j]
#
#  ATTENTION (stabilite) : le FTCS pur est INCONDITIONNELLEMENT INSTABLE pour
#  l'equation de Schrodinger. L'analyse de von Neumann donne un facteur
#  d'amplification  G = 1 - 4 i r sin^2(k dx/2)  avec  r = hbar dt/(2 m dx^2),
#  donc |G|^2 = 1 + 16 r^2 sin^4(k dx/2) > 1 pour tout dt > 0 : la norme finit
#  toujours par diverger. Borne PRATIQUE qui ralentit la divergence et garde
#  l'erreur bornee sur un temps COURT :   r <= 1/2  <=>  dt <= m dx^2 / hbar.
#  Astuce supplementaire : prendre dx pas trop petit (r ~ 1/dx^2) repousse
#  beaucoup l'explosion des modes de haute frequence.
#  Pour un calcul reellement stable et unitaire, on utiliserait :
#    - Crank-Nicolson (implicite, forme de Cayley, |G|=1 pour tout dt) ;
#    - ou le schema explicite de Visscher (parties Re/Im decalees d'un demi-pas).
#  On garde ici le FTCS pour rester "basique", en se limitant a un temps court.
#
#  Sources (consultees le 16 juin 2026) :
#   - A. Goldberg, H. M. Schey, J. L. Schwartz, "Computer-Generated Motion
#     Pictures of One-Dimensional Quantum-Mechanical Transmission and Reflection
#     Phenomena", American Journal of Physics 35, 177-186 (1967).
#     https://pubs.aip.org/aapt/ajp/article-abstract/35/3/177/1042551/Computer-Generated-Motion-Pictures-of-One
#   - "FTCS scheme", Wikipedia.  https://en.wikipedia.org/wiki/FTCS_scheme
#   - "Von Neumann Stability Analysis", Numerical Modeling Lecture Notes.
#     https://joaobuibergen.github.io/numerical-modeling-notes/NumericalStability/vonNeumannStability.html
#   - P. B. Visscher, "A fast explicit algorithm for the time-dependent
#     Schrodinger equation", Computers in Physics 5, 596 (1991).
#     https://pubs.aip.org/aip/cip/article/5/6/596/279764/A-fast-explicit-algorithm-for-the-time-dependent
#   - "Crank-Nicolson method", Wikipedia.
#     https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
# -----------------------------------------------------------------------------
#  UNITES REDUITES : on pose hbar = 1 et m = 1 (unites atomiques). En SI on
#  aurait hbar = 1.0546e-34 J.s et m_e = 9.109e-31 kg ; les nombres seraient
#  alors minuscules et le schema ingerable. Avec hbar = m = 1 tout est d'ordre 1.
#
#  But de ce programme : VERIFIER le solveur. Pour V0 = 0 (particule libre), on
#  compare le resultat numerique a la forme fermee analytique GaussWP (prog. 2).
# =============================================================================

from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Unites reduites
hbar = 1.0
m = 1.0


def GaussWP(k0, a, x, t):
    """Forme fermee analytique du paquet gaussien libre (cf. programme 2),
    ecrite ici en unites reduites (hbar = m = 1). Sert de reference exacte
    pour valider le solveur a V0 = 0. Calcul point par point (boucle for)."""
    n = len(x)
    psi = zeros(n, dtype=complex)
    prefacteur = (1.0 / (8.0 * pi**3)) ** 0.25
    for i in range(n):
        D = m * a**2 + 2j * hbar * t
        racine = sqrt(4.0 * pi * m * a / D)
        arg = (m / 4.0) * (a**2 * k0 + 2j * x[i])**2 / D - a**2 * k0**2 / 4.0
        psi[i] = prefacteur * racine * exp(arg)
    return psi


def norme_slice(psi_ligne, dx):
    """Integrale de |psi|^2 dx sur une tranche temporelle (boucle for)."""
    somme = 0.0
    for j in range(len(psi_ligne)):
        somme = somme + abs(psi_ligne[j])**2 * dx
    return somme


def resoudre_schrodinger(x, t, V, psi0):
    """Resout l'equation de Schrodinger par le schema explicite FTCS.

    x    : tableau des positions (nx points)
    t    : tableau des instants (nt points)
    V    : tableau du potentiel V[j] (taille nx)
    psi0 : condition initiale (tableau complexe taille nx)
    Renvoie le tableau 2D psi[n][j] (n = temps, j = espace).
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Verification (pedagogique) de la borne pratique de stabilite
    r = hbar * dt / (2.0 * m * dx**2)
    print("  pas d'espace dx =", dx, " pas de temps dt =", dt)
    print("  parametre r = hbar*dt/(2*m*dx^2) =", r, " (borne pratique : r <= 0.5)")
    if r > 0.5:
        print("  ATTENTION : r > 0.5, le schema FTCS va diverger tres vite !")

    # Tableau 2D : une LIGNE par instant (n), une COLONNE par point d'espace (j).
    # On alloue tout en complexe ; la premiere ligne = condition initiale.
    psi = zeros((nt, nx), dtype=complex)
    for j in range(nx):
        psi[0][j] = psi0[j]

    # Double boucle de propagation (schema explicite FTCS)
    for n in range(nt - 1):
        # bords fixes a zero (conditions de Dirichlet : paquet nul aux extremites)
        psi[n + 1][0] = 0.0
        psi[n + 1][nx - 1] = 0.0
        # points interieurs
        for j in range(1, nx - 1):
            laplacien = psi[n][j + 1] - 2.0 * psi[n][j] + psi[n][j - 1]
            psi[n + 1][j] = (psi[n][j]
                             + 1j * hbar * dt / (2.0 * m * dx**2) * laplacien
                             - 1j * dt / hbar * V[j] * psi[n][j])
    return psi


def validation_particule_libre():
    """V0 = 0 partout : on compare le solveur numerique a la solution
    analytique GaussWP, et on suit la conservation de la norme."""
    # --- parametres (unites reduites) ---
    a = 2.0           # largeur du paquet
    k0 = 2.0          # nombre d'onde central (le paquet se deplace vers la droite)
    x_min, x_max = -15.0, 15.0
    nx = 301          # -> dx = 0.1
    t_max = 1.5       # temps COURT (le FTCS ne reste valable que peu de temps)
    nt = 3001         # -> dt = 5e-4

    x = linspace(x_min, x_max, nx)
    t = linspace(0.0, t_max, nt)
    dx = x[1] - x[0]

    # Potentiel nul partout (particule libre) -- construit avec une boucle/if
    V = zeros(nx)
    for j in range(nx):
        V[j] = 0.0     # V0 = 0 pour la validation

    # Condition initiale = paquet gaussien analytique a t = 0
    psi0 = GaussWP(k0, a, x, 0.0)

    print("=== Validation du solveur a V0 = 0 (particule libre) ===")
    psi = resoudre_schrodinger(x, t, V, psi0)

    # Suivi de la norme a quelques instants
    print("  norme a t=0      :", norme_slice(psi[0], dx))
    print("  norme a t=t_max/2:", norme_slice(psi[(nt - 1) // 2], dx))
    print("  norme a t=t_max  :", norme_slice(psi[nt - 1], dx))

    # Solution analytique au temps final
    psi_ana = GaussWP(k0, a, x, t_max)

    # Densites de probabilite (boucles for)
    dens_num = zeros(nx)
    dens_ana = zeros(nx)
    for j in range(nx):
        dens_num[j] = abs(psi[nt - 1][j])**2
        dens_ana[j] = abs(psi_ana[j])**2

    # Erreur relative (norme L2) sur la densite au temps final
    num = 0.0
    den = 0.0
    for j in range(nx):
        num = num + (dens_num[j] - dens_ana[j])**2
        den = den + dens_ana[j]**2
    err_rel = sqrt(num) / sqrt(den)
    print("  erreur relative L2 sur |Psi|^2 au temps final :", err_rel)
    if err_rel < 0.05:
        print("  -> OK : le solveur reproduit bien la solution analytique.")
    else:
        print("  -> ecart notable (raffiner dx/dt ou raccourcir t_max).")

    # Trace : densites numerique vs analytique (initiale et finale)
    dens_ini = zeros(nx)
    for j in range(nx):
        dens_ini[j] = abs(psi0[j])**2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, dens_ini, "g--", label="|Psi|^2 initial (t=0)")
    ax.plot(x, dens_ana, "k-", linewidth=2, label="|Psi|^2 analytique (t_max)")
    ax.plot(x, dens_num, "r.", markersize=4, label="|Psi|^2 numerique FTCS (t_max)")
    ax.set_xlabel("position x")
    ax.set_ylabel("densite de probabilite |Psi|^2")
    ax.set_title("Validation du solveur a V0 = 0 : numerique vs analytique")
    ax.legend()
    ax.grid(True)
    plt.show()


def main():
    choix = "tout"
    if choix == "tout":
        validation_particule_libre()
    else:
        print("Choix inconnu :", choix)


if __name__ == "__main__":
    main()
