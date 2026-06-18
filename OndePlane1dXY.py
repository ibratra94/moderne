# -*- coding: utf-8 -*-
# =============================================================================
#  Projet "Effet tunnel" - Physique moderne
#  Programme 1 : ONDES PLANES (a 1 dimension)
#
#  Partie 1.1 : une onde plane unique  Psi(x,t) = A * exp(i (k x - omega t))
#  Partie 1.2 : superposition de 3 ondes planes (battements / enveloppe)
#
#  Rappels de physique (particule libre) - a savoir expliquer a l'oral :
#    - Relation de dispersion :  omega = hbar k^2 / (2 m)
#    - Vitesse de phase   :  v_phi = omega / k   = hbar k / (2 m)
#    - Vitesse de groupe  :  v_g   = d(omega)/dk = hbar k / m = p / m
#      => v_g = 2 * v_phi  (la vitesse de groupe est la vitesse "classique" p/m)
#    - Dimension de l'amplitude en 1D : [|Psi|^2] = m^-1 (densite de proba
#      par unite de longueur) donc [A] = m^(-1/2).
#    - Condition de normalisation : integrale de |Psi|^2 dx = 1
#         * particule libre : bornes -inf .. +inf  (R tout entier)
#         * puits infini    : bornes  0 .. L
#      L'onde plane a |Psi|^2 = |A|^2 = constante => integrale infinie sur R :
#      elle N'EST PAS normalisable, donc pas physiquement acceptable seule
#      (il faut superposer des ondes planes -> paquet d'ondes, programme 2).
#
#  Style : Python 3 + numpy + matplotlib uniquement (niveau L2/L3).
#  NB : "XY" dans le nom du fichier = numero de groupe de TD (X) et lettre de
#       groupe de projet (Y), a remplacer par votre binome.
# =============================================================================

# Imports de depart imposes par le sujet (+ cos pour l'enveloppe)
from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
import matplotlib.pyplot as plt


def PlaneWave(amp, k, omega, x, t):
    """Onde plane Psi(x,t) = amp * exp( i (k x - omega t) ).

    amp   : amplitude A
    k     : nombre d'onde (rad/m)
    omega : pulsation (rad/s)
    x     : position (nombre OU tableau numpy)
    t     : instant
    Renvoie un nombre (ou un tableau) complexe : 1j est l'unite imaginaire.
    """
    # exp de numpy accepte un argument complexe : pas besoin de boucle ici,
    # c'est exactement l'expression demandee par le sujet.
    return amp * exp(1j * (k * x - omega * t))


def trace_onde_unique():
    """Partie 1.1 : trace les parties reelle et imaginaire d'une onde plane
    en fonction de x, a un instant t fixe."""
    # Parametres (unites arbitraires pour cette illustration)
    amplitude = 1.0
    nombre_onde = 2.0       # k
    pulsation = 3.0         # omega
    instant = 0.0           # t fixe

    # Domaine spatial : quelques longueurs d'onde (lambda = 2 pi / k)
    longueur_onde = 2.0 * pi / nombre_onde
    x = linspace(0.0, 4.0 * longueur_onde, 600)

    # Calcul de l'onde
    psi = PlaneWave(amplitude, nombre_onde, pulsation, x, instant)

    # Trace
    fig, ax = plt.subplots()
    ax.plot(x, real(psi), label="partie reelle  Re(Psi) = A cos(kx - wt)")
    ax.plot(x, imag(psi), label="partie imaginaire  Im(Psi) = A sin(kx - wt)")
    ax.set_xlabel("position x")
    ax.set_ylabel("Psi(x, t fixe)")
    ax.set_title("Partie 1.1 : onde plane  Psi = A exp(i(kx - wt))")
    ax.legend()
    ax.grid(True)
    plt.show()


def superposition_trois_ondes(amplitude, k0, delta_k, x, t):
    """Partie 1.2 : somme de 3 ondes planes.
      - onde 1 : nombre d'onde k0        , amplitude A
      - onde 2 : nombre d'onde k0 - dk/2 , amplitude A/2
      - onde 3 : nombre d'onde k0 + dk/2 , amplitude A/2
    (toutes les pulsations sont prises a t = 0 ici, on regarde la forme en x.)

    Resultat analytique attendu (a redemontrer) :
        Psi = A exp(i k0 x) * [ 1 + cos(dk x / 2) ]
    -> porteuse exp(i k0 x), enveloppe A (1 + cos(dk x / 2)).
    On renvoie les 3 ondes, leur somme, et l'enveloppe.
    """
    onde1 = PlaneWave(amplitude, k0, 0.0, x, t)
    onde2 = PlaneWave(amplitude / 2.0, k0 - delta_k / 2.0, 0.0, x, t)
    onde3 = PlaneWave(amplitude / 2.0, k0 + delta_k / 2.0, 0.0, x, t)
    somme = onde1 + onde2 + onde3
    # Enveloppe analytique (reelle, positive)
    enveloppe = amplitude * (1.0 + cos(delta_k * x / 2.0))
    return onde1, onde2, onde3, somme, enveloppe


def trace_superposition():
    """Trace les parties reelles des 3 ondes, de leur somme, et l'enveloppe,
    sur l'intervalle [ -pi/dk , pi/dk ]."""
    amplitude = 1.0
    k0 = 4.0
    delta_k = 0.5

    # Intervalle impose par le sujet : [ -pi/dk , pi/dk ]
    x = linspace(-pi / delta_k, pi / delta_k, 1000)

    onde1, onde2, onde3, somme, enveloppe = superposition_trois_ondes(
        amplitude, k0, delta_k, x, 0.0)

    fig, ax = plt.subplots()
    # Les 3 ondes (parties reelles), en traits fins
    ax.plot(x, real(onde1), linewidth=0.8, label="Re(onde 1)  k0")
    ax.plot(x, real(onde2), linewidth=0.8, label="Re(onde 2)  k0 - dk/2")
    ax.plot(x, real(onde3), linewidth=0.8, label="Re(onde 3)  k0 + dk/2")
    # La somme, en trait epais
    ax.plot(x, real(somme), color="black", linewidth=1.8, label="Re(somme)")
    # L'enveloppe + et - (en pointilles)
    ax.plot(x, enveloppe, "r--", label="enveloppe  A(1 + cos(dk x/2))")
    ax.plot(x, -enveloppe, "r--")

    ax.set_xlabel("position x")
    ax.set_ylabel("partie reelle")
    ax.set_title("Partie 1.2 : superposition de 3 ondes planes et enveloppe")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    plt.show()


def main():
    # Petit menu "procedural" : choisir la partie a executer.
    # Mettre choix = "1.1", "1.2" ou "tout".
    choix = "tout"

    if choix == "1.1":
        trace_onde_unique()
    elif choix == "1.2":
        trace_superposition()
    elif choix == "tout":
        print("Partie 1.1 : onde plane unique")
        trace_onde_unique()
        print("Partie 1.2 : superposition de 3 ondes planes")
        trace_superposition()
    else:
        print("Choix inconnu :", choix)


if __name__ == "__main__":
    main()
