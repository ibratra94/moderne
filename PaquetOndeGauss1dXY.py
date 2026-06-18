# -*- coding: utf-8 -*-
# =============================================================================
#  Projet "Effet tunnel" - Physique moderne
#  Programme 2 : PAQUET D'ONDES GAUSSIEN (a 1 dimension)
#
#  Un paquet d'ondes est une superposition CONTINUE d'ondes planes (integrale
#  sur k). Si la fonction de repartition g(k) est une gaussienne :
#        g(k) = sqrt(a) (2 pi)^(-1/4) exp( -a^2 (k - k0)^2 / 4 )
#  alors on connait une FORME FERMEE du paquet a tout instant t (resultat
#  fourni par le sujet, implemente dans la fonction GaussWP ci-dessous).
# -----------------------------------------------------------------------------
#  Paquet d'ondes gaussien libre : forme fermee Psi(x,t)
#    Psi(x,t) = (1/(8 pi^3))^(1/4) * sqrt(4 pi m a / (m a^2 + 2 i hbar t))
#             * exp[ (m/4)(a^2 k0 + 2 i x)^2 / (m a^2 + 2 i hbar t) - a^2 k0^2/4 ]
#
#  Verifications (faites analytiquement + numeriquement) :
#    - a t=0 : Psi(x,0) = (2/pi)^(1/4) a^(-1/2) exp(i k0 x) exp(-x^2/a^2)   [OK]
#    - normalisation a t=0 : integrale de |Psi|^2 dx = 1                    [OK]
#    - racine complexe : Re(m a^2 + 2 i hbar t) = m a^2 > 0 pour tout t,
#      donc sqrt (branche principale de numpy) est correcte et continue.
#      ATTENTION : garder TOUT le quotient sous UN seul sqrt, ne jamais
#      separer en sqrt(num)/sqrt(den) (risque de saut de branche).
#
#  Sources (consultees le 16 juin 2026) :
#    - N. Walet, "Wave packets (states of minimal uncertainty)",
#      Quantum Mechanics, Physics LibreTexts, Univ. of Manchester.
#      https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Quantum_Mechanics_(Walet)/10:_Time-Dependent_Wavefunctions/10.05:_Wave_packets_(states_of_minimal_uncertainty)
#    - S. Golwala, "The Free Particle: Gaussian Wave Packets", notes Ph125ab,
#      California Institute of Technology.
#      https://sites.astro.caltech.edu/~golwala/ph125ab/ph125_notes_l15.pdf
#    - N. Wheeler, "Gaussian Wavepackets", Reed College Physics Department.
#      https://www.reed.edu/physics/faculty/wheeler/documents/Quantum%20Mechanics/Miscellaneous%20Essays/Gaussian%20Wavepackets.pdf
# -----------------------------------------------------------------------------
#  DIFFICULTE RENCONTREE (echelles physiques) :
#    Avec la masse de l'electron en unites SI, hbar ~ 1e-34 et m ~ 9.1e-31 sont
#    minuscules. Pour que le paquet ait quelques oscillations sous l'enveloppe,
#    il faut une largeur a de l'ordre du nanometre (a ~ 1e-9 m) et un nombre
#    d'onde k0 ~ 1e10 m^-1 (energie de quelques eV). Les axes du graphe sont
#    alors a des echelles tres differentes (x ~ 1e-9 m, Psi ~ 1e4 m^(-1/2)).
#    ASTUCE retenue :
#      (a) on trace x en nanometres pour la lisibilite ;
#      (b) pour l'EVOLUTION TEMPORELLE (programmes 3 et 4), on passe en UNITES
#          REDUITES (hbar = 1, m = 1, "unites atomiques") : tous les nombres
#          redeviennent d'ordre 1, ce qui rend le schema numerique gerable.
# =============================================================================

# Imports de depart imposes par le sujet
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Constantes physiques (Systeme International) - electron
hbar = 1.054571817e-34   # constante de Planck reduite (J.s)
m = 9.1093837015e-31     # masse de l'electron (kg)


def GaussWP(k0, a, x, t):
    """Paquet d'ondes gaussien libre, forme fermee Psi(x,t).

    k0 : nombre d'onde central (m^-1)
    a  : parametre de largeur du paquet (m)   [a t=0 : exp(-x^2/a^2)]
    x  : tableau des positions (m)
    t  : instant (s)
    Renvoie le tableau complexe Psi(x, t).

    On remplit le tableau point par point avec une boucle for (style etudiant),
    sans vectorisation maligne. sqrt et exp de numpy acceptent les complexes.
    """
    n = len(x)
    psi = zeros(n, dtype=complex)

    # Prefacteur (1 / (8 pi^3))^(1/4) : reel positif, calcule une seule fois.
    prefacteur = (1.0 / (8.0 * pi**3)) ** 0.25

    for i in range(n):
        # Denominateur complexe commun D = m a^2 + 2 i hbar t
        D = m * a**2 + 2j * hbar * t

        # Racine complexe : TOUT le quotient sous un seul sqrt (cf. en-tete)
        racine = sqrt(4.0 * pi * m * a / D)

        # Argument de l'exponentielle
        arg = (m / 4.0) * (a**2 * k0 + 2j * x[i])**2 / D - a**2 * k0**2 / 4.0

        psi[i] = prefacteur * racine * exp(arg)

    return psi


def norme(psi, dx):
    """Calcule integrale de |psi|^2 dx par une somme de Riemann (boucle for)."""
    somme = 0.0
    for i in range(len(psi)):
        somme = somme + abs(psi[i])**2 * dx
    return somme


def trace_paquet_t0():
    """Trace les parties reelle et imaginaire (et la densite |Psi|^2) du paquet
    a l'instant t = 0, et verifie la normalisation."""
    # Parametres physiques (echelle nanometrique, cf. en-tete)
    a = 1.0e-9          # largeur du paquet (1 nm)
    k0 = 1.0e10         # nombre d'onde central (m^-1) -> energie ~ 3.8 eV
    t = 0.0

    # Domaine spatial : +- 5 a autour du centre (le paquet est centre en 0)
    x = linspace(-5.0 * a, 5.0 * a, 1000)
    dx = x[1] - x[0]

    # Calcul du paquet
    psi = GaussWP(k0, a, x, t)

    # Verification de la normalisation : integrale |Psi|^2 dx doit valoir ~ 1
    n2 = norme(psi, dx)
    print("Verification de la norme a t=0 : integrale |Psi|^2 dx =", n2)
    if abs(n2 - 1.0) < 1.0e-3:
        print("  -> OK, le paquet gaussien est bien normalise (= 1).")
    else:
        print("  -> ATTENTION : norme eloignee de 1 (elargir le domaine x ?).")

    # Densite de probabilite |Psi|^2 (boucle for)
    densite = zeros(len(x))
    for i in range(len(x)):
        densite[i] = abs(psi[i])**2

    # On trace x en nanometres pour la lisibilite (cf. ASTUCE en-tete)
    x_nm = x * 1.0e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    ax1.plot(x_nm, real(psi), label="Re(Psi)")
    ax1.plot(x_nm, imag(psi), label="Im(Psi)")
    ax1.set_xlabel("position x (nm)")
    ax1.set_ylabel("Psi  (m^-1/2)")
    ax1.set_title("Paquet d'ondes gaussien a t = 0 : parties reelle et imaginaire")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_nm, densite, color="black")
    ax2.set_xlabel("position x (nm)")
    ax2.set_ylabel("|Psi|^2  (m^-1)")
    ax2.set_title("Densite de probabilite |Psi|^2 (aire = 1)")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


def trace_etalement():
    """Illustration (bonus) : on profite de la dependance en t de GaussWP pour
    montrer que le paquet libre se DEPLACE (a la vitesse de groupe v_g = hbar k0/m)
    et S'ETALE au cours du temps. C'est le comportement decrit en partie 2."""
    a = 1.0e-9
    k0 = 1.0e10
    # domaine elargi car le paquet avance de ~11 nm en 1e-14 s
    x = linspace(-20.0e-9, 20.0e-9, 1500)
    dx = x[1] - x[0]

    instants = [0.0, 0.5e-14, 1.0e-14]   # quelques instants (en secondes)

    fig, ax = plt.subplots(figsize=(9, 5))
    for t in instants:
        psi = GaussWP(k0, a, x, t)
        densite = zeros(len(x))
        for i in range(len(x)):
            densite[i] = abs(psi[i])**2
        ax.plot(x * 1.0e9, densite, label="t = %.1e s" % t)
        print("  t = %.2e s : norme =" % t, norme(psi, dx),
              " (doit rester ~ 1 : la norme se conserve dans le temps)")
    ax.set_xlabel("position x (nm)")
    ax.set_ylabel("|Psi|^2  (m^-1)")
    ax.set_title("Le paquet gaussien libre se deplace (v_g) et s'etale")
    ax.legend()
    ax.grid(True)
    plt.show()


def main():
    # Menu procedural simple : "t0", "etalement" ou "tout".
    choix = "tout"

    if choix == "t0":
        trace_paquet_t0()
    elif choix == "etalement":
        trace_etalement()
    elif choix == "tout":
        trace_paquet_t0()
        trace_etalement()
    else:
        print("Choix inconnu :", choix)


if __name__ == "__main__":
    main()
