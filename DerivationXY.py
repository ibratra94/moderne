# -*- coding: utf-8 -*-
# =============================================================================
#  Projet "Effet tunnel" - Physique moderne
#  Programme 3 (partie A) : DERIVATION NUMERIQUE par DIFFERENCES FINIES
#
#  Rappel : la derivee est la limite du taux d'accroissement
#        f'(x) = lim_{h->0} ( f(x+h) - f(x) ) / h
#  Sur un tableau de valeurs echantillonnees au pas dx, on approche :
# -------------------------------------------------------------------------
#  Derivees premiere et seconde par la methode des differences finies (1D)
#  Schemas centres a l'interieur (ordre 2), decentres aux bords.
#    f'(x_j)  ~ (f[j+1] - f[j-1]) / (2*dx)          (centre, O(dx^2))
#    f''(x_j) ~ (f[j+1] - 2*f[j] + f[j-1]) / dx^2   (centre, O(dx^2))
#    bords : differences avant / arriere.
#
#  Sources (consultees le 16 juin 2026) :
#   - M. Buffat, "Approximation des derivees par differences finies",
#     cours de calcul scientifique, Universite Claude Bernard Lyon 1.
#     https://perso.univ-lyon1.fr/marc.buffat/COURS/COURSDF_HTML/node16.html
#   - "Finite difference coefficient", Wikipedia (tableaux de coefficients).
#     https://en.wikipedia.org/wiki/Finite_difference_coefficient
#   - J. R. Chasnov, "6: Finite Difference Approximation", Scientific
#     Computing, Mathematics LibreTexts.
#     https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Scientific_Computing_(Chasnov)/I:_Numerical_Methods/6:_Finite_Difference_Approximation
# -------------------------------------------------------------------------
#  Style : numpy + matplotlib, boucles for explicites, if/elif pour les bords.
# =============================================================================

from numpy import pi, exp, sqrt, real, imag, zeros, linspace, sin, cos
import matplotlib.pyplot as plt


# ---------- fonctions de test : f(x) = x^2  et sa derivee exacte 2x ----------
def carre(x):
    """f(x) = x^2"""
    return x * x


def deux_x(x):
    """f'(x) = 2 x  (derivee premiere exacte attendue)"""
    return 2.0 * x


def deux_constant(x):
    """f''(x) = 2  (derivee seconde exacte attendue, constante)"""
    return 2.0 + 0.0 * x   # 0*x pour renvoyer un tableau de la bonne taille


# ----------------------------- derivee premiere ------------------------------
def derivee_premiere(f, dx):
    """Derivee premiere de f (tableau) au pas dx, par differences finies.
    - interieur : difference centree (ordre 2)
    - bords     : difference avant (gauche) / arriere (droite), ordre 1
    """
    n = len(f)
    df = zeros(n)
    for i in range(n):
        if i == 0:
            # bord gauche : difference avant
            df[i] = (f[i + 1] - f[i]) / dx
        elif i == n - 1:
            # bord droit : difference arriere
            df[i] = (f[i] - f[i - 1]) / dx
        else:
            # interieur : difference centree
            df[i] = (f[i + 1] - f[i - 1]) / (2.0 * dx)
    return df


# ----------------------------- derivee seconde -------------------------------
def derivee_seconde(f, dx):
    """Derivee seconde de f (tableau) au pas dx, par differences finies.
    - interieur : difference centree a 3 points (ordre 2)
    - bords     : formule decentree a 3 points (ordre 1)
    """
    n = len(f)
    d2f = zeros(n)
    for i in range(n):
        if i == 0:
            # bord gauche : formule avant a 3 points
            d2f[i] = (f[i] - 2.0 * f[i + 1] + f[i + 2]) / dx**2
        elif i == n - 1:
            # bord droit : formule arriere a 3 points
            d2f[i] = (f[i] - 2.0 * f[i - 1] + f[i - 2]) / dx**2
        else:
            # interieur : difference centree
            d2f[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / dx**2
    return d2f


# --------------------------- outil : erreur relative -------------------------
def erreur_relative_max(approx, exact):
    """Plus grande erreur relative |approx - exact| / |exact| sur le tableau,
    en evitant la division par zero (on ignore les points ou exact ~ 0)."""
    err_max = 0.0
    for i in range(len(approx)):
        if abs(exact[i]) > 1.0e-12:
            err = abs(approx[i] - exact[i]) / abs(exact[i])
            if err > err_max:
                err_max = err
    return err_max


# --------------------------------- tests -------------------------------------
def test_sur_x_carre():
    """Test impose : deriver f(x) = x^2 et comparer a f'(x)=2x et f''(x)=2."""
    # On evite x = 0 pour pouvoir calculer une erreur RELATIVE propre.
    x = linspace(1.0, 3.0, 41)
    dx = x[1] - x[0]

    f = carre(x)
    df_num = derivee_premiere(f, dx)
    d2f_num = derivee_seconde(f, dx)

    df_exact = deux_x(x)
    d2f_exact = deux_constant(x)

    print("=== Test sur f(x) = x^2 (dx =", dx, ") ===")
    # Remarque pedagogique : la difference CENTREE est EXACTE pour un polynome
    # de degre 2 (l'erreur de troncature ~ f''' s'annule pour x^2). On observe
    # donc une erreur quasi nulle a l'interieur, et une erreur ~ dx aux bords
    # (ou l'on utilise une formule decentree d'ordre 1).
    print("  derivee premiere : erreur relative max (tout le domaine) =",
          erreur_relative_max(df_num, df_exact))
    print("  derivee premiere : erreur relative max (interieur seul)  =",
          erreur_relative_max(df_num[1:-1], df_exact[1:-1]))
    print("  derivee seconde  : erreur relative max =",
          erreur_relative_max(d2f_num, d2f_exact))

    # Trace comparatif
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(x, df_exact, "k-", label="exact : 2x")
    ax1.plot(x, df_num, "ro", markersize=3, label="numerique")
    ax1.set_title("Derivee premiere de x^2")
    ax1.set_xlabel("x"); ax1.set_ylabel("f'(x)"); ax1.legend(); ax1.grid(True)

    ax2.plot(x, d2f_exact, "k-", label="exact : 2")
    ax2.plot(x, d2f_num, "bo", markersize=3, label="numerique")
    ax2.set_title("Derivee seconde de x^2")
    ax2.set_xlabel("x"); ax2.set_ylabel("f''(x)"); ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    plt.show()


def test_convergence_sinus():
    """Test complementaire : f(x) = sin(x). Comme sin n'est pas un polynome,
    on observe la VRAIE erreur des schemas centres et sa decroissance en
    O(dx^2) quand on raffine le maillage (on divise dx par 2 -> erreur / 4)."""
    print("=== Test de convergence sur f(x) = sin(x) ===")
    print("  (l'erreur centree doit etre divisee par ~4 quand dx est divise par 2)")
    for nb_points in [21, 41, 81, 161]:
        x = linspace(0.0, pi, nb_points)
        dx = x[1] - x[0]
        f = sin(x)
        df_num = derivee_premiere(f, dx)
        df_exact = cos(x)            # derivee exacte de sin
        # erreur a l'interieur (on enleve les bords d'ordre 1)
        err = erreur_relative_max(df_num[1:-1], df_exact[1:-1])
        print("  nb_points =", nb_points, " dx =", round(dx, 5),
              " erreur relative interieure =", err)


def main():
    # Menu procedural simple.
    choix = "tout"

    if choix == "x2":
        test_sur_x_carre()
    elif choix == "sinus":
        test_convergence_sinus()
    elif choix == "tout":
        test_sur_x_carre()
        test_convergence_sinus()
    else:
        print("Choix inconnu :", choix)


if __name__ == "__main__":
    main()
