# -*- coding: utf-8 -*-
# =============================================================================
#  Projet "Effet tunnel" - Physique moderne
#  Programme 4 : LE PROJET - effet tunnel d'un paquet d'ondes a travers une
#                barriere de potentiel rectangulaire (resolution numerique).
# -----------------------------------------------------------------------------
#  On ajoute une barriere rectangulaire de hauteur V0 et de largeur a, et on
#  mesure des "temps de franchissement" + l'influence de a et de V0.
#
#  CHOIX DU SCHEMA NUMERIQUE :
#  Le schema explicite FTCS du programme 3 est INSTABLE (la norme diverge). Pour
#  une barriere, ses bords nets excitent des composantes de haute frequence que
#  le FTCS amplifie tres vite : la simulation EXPLOSE avant la fin (on l'a
#  constate : norme -> 1e5 !). On utilise donc ici le schema explicite STABLE de
#  P. B. Visscher (1991), deja cite dans le programme 3 : on ecrit psi = R + i I
#  et on calcule R aux pas de temps entiers et I aux pas demi-entiers (schema
#  "saute-mouton" / leapfrog). Il reste tres simple (boucles for, numpy seul,
#  pas de scipy), il est du 2e ordre en temps et il CONSERVE la norme.
#    Equations (psi = R + iI, H = -(hbar^2/2m) d2/dx2 + V) :
#       dR/dt = +H[I]      dI/dt = -H[R]
#    Densite (forme symetrique de Visscher, bien conservee) :
#       rho^n = (R^n)^2 + I^{n-1/2} * I^{n+1/2}
#    Stabilite (conditionnelle) :  dt <= hbar / ( hbar^2/(m dx^2) + V_max/2 ).
#  Source : P. B. Visscher, "A fast explicit algorithm for the time-dependent
#  Schrodinger equation", Computers in Physics 5, 596 (1991).
#  https://pubs.aip.org/aip/cip/article/5/6/596/279764/A-fast-explicit-algorithm-for-the-time-dependent
#
#  Formules analytiques de reference (barriere rectangulaire, MQ 1D) - verifiees :
#    - Effet tunnel 0 < E < V0 :  kappa = sqrt(2 m (V0 - E)) / hbar
#        T = [ 1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E)) ]^(-1)
#    - Cas E > V0 :  k' = sqrt(2 m (E - V0)) / hbar
#        T = [ 1 + V0^2 sin^2(k' a) / (4 E (E - V0)) ]^(-1)   (resonances)
#    - Particule libre :  v_g = hbar k0 / m  (vitesse de groupe = vitesse du paquet)
#    - Temps de traversee theorique (libre) :  tau0_th = a / v_g = a m / (hbar k0)
#
#  Sources (consultees le 16 juin 2026) :
#    - A. Goldberg, H. M. Schey, J. L. Schwartz, "Computer-Generated Motion
#      Pictures of One-Dimensional Quantum-Mechanical Transmission and Reflection
#      Phenomena", American Journal of Physics 35, 177-186 (1967). [ref. fondatrice]
#      https://ui.adsabs.harvard.edu/abs/1967AmJPh..35..177G/abstract
#    - Wikipedia, "Rectangular potential barrier".
#      https://en.wikipedia.org/wiki/Rectangular_potential_barrier
#    - OpenStax / LibreTexts, University Physics III, sec. 7.7,
#      "Quantum Tunneling of Particles through Potential Barriers".
#      https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/07:_Quantum_Mechanics/7.07:_Quantum_Tunneling_of_Particles_through_Potential_Barriers
#    - Ref. ouvrage : D. J. Griffiths, "Introduction to Quantum Mechanics".
# -----------------------------------------------------------------------------
#  REMARQUE IMPORTANTE sur le "temps tunnel" : il n'existe PAS de definition
#  unique du temps que met une particule pour franchir une barriere (c'est un
#  sujet de recherche encore ouvert). On adopte ici une definition OPERATIONNELLE
#  simple, que l'on assume :
#    - tau0_num : temps que met le CENTRE <x> du paquet LIBRE pour parcourir une
#      distance egale a la largeur a de la barriere (doit valoir a / v_g).
#    - "retard" tunnel : on place un detecteur fixe a droite de la barriere et on
#      compare l'instant d'arrivee du centre du paquet TRANSMIS a celui du paquet
#      LIBRE. retard = t_transmis - t_libre. C'est notre indicateur operationnel
#      de temps de franchissement (souvent ~0 ou negatif : cf. filtrage en
#      energie + effet Hartman, plus bas).
#
#  REMARQUE sur le stockage : contrairement au programme 3 (tableau 2D complet),
#  on ne garde ici que la tranche temporelle COURANTE (deux tableaux psi/psi_new)
#  plus quelques "images" pour les graphes : sinon il faudrait stocker des
#  millions de nombres complexes (nt*nx) pour chaque simulation.
#
#  UNITES REDUITES : hbar = 1, m = 1 (cf. programmes 2 et 3).
# =============================================================================

from numpy import pi, exp, sqrt, real, imag, zeros, linspace, sin, sinh
import matplotlib.pyplot as plt

# Unites reduites
hbar = 1.0
m = 1.0


# ---------------------------------------------------------------------------
#  Construction du paquet d'ondes initial (gaussien) centre en x0, allant vers
#  la droite (k0 > 0). On normalise numeriquement : integrale |psi|^2 dx = 1.
# ---------------------------------------------------------------------------
def construire_paquet(x, x0, a, k0):
    nx = len(x)
    dx = x[1] - x[0]
    psi = zeros(nx, dtype=complex)
    for j in range(nx):
        enveloppe = exp(-((x[j] - x0)**2) / a**2)   # gaussienne centree en x0
        porteuse = exp(1j * k0 * x[j])              # onde plane (vitesse > 0)
        psi[j] = enveloppe * porteuse
    # normalisation numerique
    somme = 0.0
    for j in range(nx):
        somme = somme + abs(psi[j])**2 * dx
    facteur = 1.0 / sqrt(somme)
    for j in range(nx):
        psi[j] = psi[j] * facteur
    return psi


# ---------------------------------------------------------------------------
#  Construction de la barriere rectangulaire : V = V0 entre debut et debut+larg,
#  V = 0 ailleurs. Conditions explicites (if/elif/else), comme demande.
# ---------------------------------------------------------------------------
def construire_barriere(x, debut, largeur, V0):
    nx = len(x)
    V = zeros(nx)
    fin = debut + largeur
    for j in range(nx):
        if x[j] < debut:
            V[j] = 0.0          # avant la barriere : particule libre
        elif x[j] <= fin:
            V[j] = V0           # dans la barriere
        else:
            V[j] = 0.0          # apres la barriere : particule libre
    return V


# ---------------------------------------------------------------------------
#  Operateur hamiltonien applique a un tableau reel f (unites reduites) :
#       H[f]_j = -(1/2) (f[j+1] - 2 f[j] + f[j-1]) / dx^2  +  V[j] f[j]
#  (hbar = m = 1). Renvoie un tableau, calcule par une boucle for ; les bords
#  restent a zero (conditions de Dirichlet).
# ---------------------------------------------------------------------------
def applique_H(f, V, dx):
    nx = len(f)
    Hf = zeros(nx)
    for j in range(1, nx - 1):
        cinetique = -0.5 * (f[j + 1] - 2.0 * f[j] + f[j - 1]) / dx**2
        Hf[j] = cinetique + V[j] * f[j]
    return Hf


# ---------------------------------------------------------------------------
#  Solveur : schema explicite STABLE de Visscher (cf. en-tete). On suit des
#  grandeurs physiques au cours du temps et on ne stocke que la tranche
#  courante (R, I) + quelques images de la densite pour les graphes.
#  psi = R + i I : R aux pas entiers n, I aux pas demi-entiers n+1/2.
#  Renvoie : temps, norme(t), <x>(t), P_transmise(t), images(densite), temps_images.
# ---------------------------------------------------------------------------
def simuler(x, V, psi0, dt, nt, x_sortie, nb_images=6):
    nx = len(x)
    dx = x[1] - x[0]

    # Parties reelle et imaginaire de la condition initiale.
    R = zeros(nx)
    I = zeros(nx)
    for j in range(nx):
        R[j] = real(psi0[j])
        I[j] = imag(psi0[j])
    # I est cense etre pris a t = -dt/2 ; on l'initialise a I(0) (erreur O(dt)).

    temps = zeros(nt)
    norme_serie = zeros(nt)
    xmoy_serie = zeros(nt)
    ptrans_serie = zeros(nt)
    xtrans_serie = zeros(nt)   # position moyenne de la SEULE partie transmise

    images = []
    temps_images = []
    if nb_images > 1:
        pas_image = (nt - 1) // (nb_images - 1)
    else:
        pas_image = nt

    for n in range(nt):
        # --- on avance I d'un demi-pas : I^{n+1/2} = I^{n-1/2} - dt H[R^n] ---
        HR = applique_H(R, V, dx)
        I_suivant = zeros(nx)
        for j in range(1, nx - 1):
            I_suivant[j] = I[j] - dt * HR[j]
        # (bords laisses a zero : I_suivant[0] = I_suivant[nx-1] = 0)

        # --- densite symetrique de Visscher : rho = R^2 + I^{n-1/2} I^{n+1/2} ---
        nrm = 0.0
        somme_x = 0.0
        proba_droite = 0.0
        somme_x_droite = 0.0
        for j in range(nx):
            densite = R[j] * R[j] + I[j] * I_suivant[j]
            nrm = nrm + densite * dx
            somme_x = somme_x + x[j] * densite * dx
            if x[j] >= x_sortie:
                proba_droite = proba_droite + densite * dx
                somme_x_droite = somme_x_droite + x[j] * densite * dx
        temps[n] = n * dt
        norme_serie[n] = nrm
        xmoy_serie[n] = somme_x / nrm          # valeur moyenne <x> (tout le paquet)
        ptrans_serie[n] = proba_droite / nrm   # fraction transmise (normalisee)
        # position moyenne de la partie transmise (si elle existe deja)
        if proba_droite > 1.0e-6:
            xtrans_serie[n] = somme_x_droite / proba_droite
        else:
            xtrans_serie[n] = x_sortie         # rien de transmis encore

        # --- enregistrement d'une image de la densite de temps en temps ---
        if pas_image > 0 and n % pas_image == 0:
            image = zeros(nx)
            for j in range(nx):
                image[j] = R[j] * R[j] + I[j] * I_suivant[j]
            images.append(image)
            temps_images.append(n * dt)

        # --- on avance R d'un pas : R^{n+1} = R^n + dt H[I^{n+1/2}] ---
        if n < nt - 1:
            HI = applique_H(I_suivant, V, dx)
            R_suivant = zeros(nx)
            for j in range(1, nx - 1):
                R_suivant[j] = R[j] + dt * HI[j]
            # bascule pour le pas suivant
            for j in range(nx):
                R[j] = R_suivant[j]
                I[j] = I_suivant[j]

    return temps, norme_serie, xmoy_serie, ptrans_serie, xtrans_serie, images, temps_images


# ---------------------------------------------------------------------------
#  Outil : premier instant ou un signal(t) atteint un seuil (interpolation
#  lineaire entre deux pas). Renvoie None si le seuil n'est jamais atteint.
# ---------------------------------------------------------------------------
def temps_de_passage(temps, signal, seuil):
    for i in range(1, len(signal)):
        if (signal[i - 1] < seuil) and (signal[i] >= seuil):
            # interpolation lineaire entre i-1 et i
            t1, t2 = temps[i - 1], temps[i]
            s1, s2 = signal[i - 1], signal[i]
            return t1 + (seuil - s1) * (t2 - t1) / (s2 - s1)
    return None


# ---------------------------------------------------------------------------
#  Coefficient de transmission analytique (onde plane d'energie E) pour la
#  barriere (V0, largeur a). if/elif/else pour les trois regimes physiques.
# ---------------------------------------------------------------------------
def transmission_analytique(E, V0, a):
    if E < V0:
        # effet tunnel
        kappa = sqrt(2.0 * m * (V0 - E)) / hbar
        sh = sinh(kappa * a)
        return 1.0 / (1.0 + (V0**2 * sh * sh) / (4.0 * E * (V0 - E)))
    elif E > V0:
        # passage classique, avec resonances
        kp = sqrt(2.0 * m * (E - V0)) / hbar
        s = sin(kp * a)
        return 1.0 / (1.0 + (V0**2 * s * s) / (4.0 * E * (E - V0)))
    else:
        # cas limite E = V0
        return 1.0 / (1.0 + m * V0 * a * a / (2.0 * hbar**2))


# ===========================================================================
#  Parametres communs a toutes les simulations (unites reduites)
# ===========================================================================
A_PAQUET = 4.0         # largeur du paquet d'ondes (assez large => bande de k
                       #   etroite, pour bien definir l'energie E = hbar^2 k0^2/2m)
K0 = 2.0               # nombre d'onde central -> energie E = hbar^2 k0^2 / 2m
X0 = -13.0             # position initiale du centre du paquet (a gauche)
X_MIN, X_MAX = -30.0, 30.0
NX = 301               # -> dx = 0.2
T_MAX = 12.0
NT = 3001              # -> dt = 4e-3 (schema de Visscher : largement stable)
DEBUT_BARRIERE = 0.0   # la barriere commence en x = 0

E0 = hbar**2 * K0**2 / (2.0 * m)   # energie (centrale) du paquet


def grille():
    x = linspace(X_MIN, X_MAX, NX)
    dt = T_MAX / (NT - 1)
    return x, dt


# ---------------------------------------------------------------------------
#  PARTIE A : visualisation de l'effet tunnel (une barriere, E < V0).
# ---------------------------------------------------------------------------
def partie_A_visualisation():
    x, dt = grille()
    V0 = 3.0           # E0 = 2 < V0 = 3 : effet tunnel, mais transmission assez
                       #   grande pour que le paquet transmis soit VISIBLE
                       #   (T_num du paquet ~ 13 % ; T analytique onde plane ~ 19 %).
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur

    print("=== Partie A : visualisation de l'effet tunnel ===")
    print("  energie du paquet E0 =", E0, " hauteur V0 =", V0,
          " (E0 < V0 : classiquement REFLECHI)")

    V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    temps, norme_s, xmoy_s, ptrans_s, xtrans_s, images, t_images = simuler(
        x, V, psi0, dt, NT, fin, nb_images=6)

    T_num = ptrans_s[NT - 1]
    T_th = transmission_analytique(E0, V0, largeur)
    print("  transmission numerique T_num (du paquet) =", T_num)
    print("  transmission analytique (onde plane a k0) T_th =", T_th)
    print("  norme finale (controle stabilite Visscher) :", norme_s[NT - 1])

    # Trace de quelques images de |psi|^2, et de la barriere (mise a l'echelle)
    fig, ax = plt.subplots(figsize=(9, 5))
    hauteur_max = 0.0
    for image in images:
        for j in range(len(x)):
            if image[j] > hauteur_max:   # image[j] est deja la densite |psi|^2
                hauteur_max = image[j]
    # barriere dessinee en gris (echelle arbitraire pour la lisibilite)
    barriere_dessin = zeros(len(x))
    for j in range(len(x)):
        if V[j] > 0:
            barriere_dessin[j] = hauteur_max
    ax.fill_between(x, 0, barriere_dessin, color="lightgray",
                    label="barriere (V0=%.1f, a=%.1f)" % (V0, largeur))
    for k in range(len(images)):
        ax.plot(x, images[k], label="t = %.2f" % t_images[k])
    ax.set_xlabel("position x")
    ax.set_ylabel("|Psi|^2")
    ax.set_title("Effet tunnel : le paquet frappe la barriere et se scinde "
                 "(reflechi + transmis)")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
#  PARTIE B : mesure des temps tau0 (libre) et taut (avec barriere).
# ---------------------------------------------------------------------------
def partie_B_temps():
    x, dt = grille()
    V0 = 4.0
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur
    x_det = 7.0                # detecteur fixe ; doit etre A DROITE de la barriere
                               #   (x_det > fin) pour que la mesure ait un sens.

    print("=== Partie B : mesure des temps de franchissement ===")

    # --- run LIBRE (V0 = 0) : mesure de tau0 et du temps d'arrivee au detecteur ---
    V_libre = construire_barriere(x, DEBUT_BARRIERE, largeur, 0.0)  # V = 0 partout
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    t_l, norme_l, xmoy_l, ptrans_l, xtrans_l, img_l, ti_l = simuler(
        x, V_libre, psi0, dt, NT, fin, nb_images=2)

    # tau0 = temps pour que <x> passe de l'entree (x=0) a la sortie (x=largeur)
    t_entree_libre = temps_de_passage(t_l, xmoy_l, DEBUT_BARRIERE)
    t_sortie_libre = temps_de_passage(t_l, xmoy_l, fin)
    tau0_th = largeur * m / (hbar * K0)        # = a / v_g
    # garde-fou : temps_de_passage renvoie None si le seuil n'est pas atteint
    if (t_entree_libre is None) or (t_sortie_libre is None):
        tau0_num = float("nan")
        print("  tau0_num non mesurable (seuils non atteints sur la fenetre de temps)")
    else:
        tau0_num = t_sortie_libre - t_entree_libre
        print("  tau0_num (paquet libre, parcours de a) =", tau0_num)
    print("  tau0_th  = a m /(hbar k0) = a / v_g    =", tau0_th)

    # temps d'arrivee du centre du paquet LIBRE au detecteur x_det
    t_arr_libre = temps_de_passage(t_l, xmoy_l, x_det)

    # --- run AVEC BARRIERE : transmission + temps d'arrivee du paquet transmis ---
    V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
    psi0b = construire_paquet(x, X0, A_PAQUET, K0)
    t_b, norme_b, xmoy_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
        x, V, psi0b, dt, NT, fin, nb_images=2)

    T_num = ptrans_b[NT - 1]
    # temps d'arrivee du centre de la partie TRANSMISE au meme detecteur
    t_arr_trans = temps_de_passage(t_b, xtrans_b, x_det)
    print("  fraction transmise finale T_num =", T_num)
    print("  arrivee au detecteur x_det =", x_det)
    print("    paquet libre    : t =", t_arr_libre)
    print("    paquet transmis : t =", t_arr_trans)
    if (t_arr_trans is not None) and (t_arr_libre is not None):
        retard = t_arr_trans - t_arr_libre
        print("  retard du paquet transmis (t_trans - t_libre) =", retard)
        if retard < 0:
            print("  => le paquet TRANSMIS arrive AVANT le paquet libre.")
    print("  INTERPRETATION : la barriere se comporte comme un FILTRE EN ENERGIE :")
    print("    elle transmet preferentiellement les composantes de haute energie")
    print("    (k > k0), qui vont plus vite ; le paquet transmis est donc plus")
    print("    rapide que le paquet libre moyen -> il peut arriver en avance.")
    print("    A cela s'ajoute le fait que le temps passe DANS la barriere est")
    print("    anormalement court (effet Hartman). Conclusion : le 'temps tunnel'")
    print("    n'est PAS a/v et reste un probleme ouvert -> mesure operationnelle,")
    print("    a interpreter avec prudence.")

    # Trace des grandeurs suivies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(t_l, xmoy_l, label="<x>(t) paquet libre")
    ax1.plot(t_b, xmoy_b, label="<x>(t) avec barriere (tout le paquet)")
    ax1.plot(t_b, xtrans_b, "g--", label="<x>(t) de la partie transmise")
    ax1.axhline(DEBUT_BARRIERE, color="gray", linestyle=":")
    ax1.axhline(fin, color="gray", linestyle=":")
    ax1.axhline(x_det, color="red", linestyle=":", label="detecteur x_det")
    ax1.set_xlabel("temps t"); ax1.set_ylabel("position moyenne <x>")
    ax1.set_title("Suivi du centre du paquet")
    ax1.legend(fontsize=8); ax1.grid(True)

    ax2.plot(t_b, ptrans_b, color="green",
             label="P transmise(t) (a droite de la barriere)")
    ax2.axhline(T_num, color="green", linestyle=":")
    ax2.set_xlabel("temps t"); ax2.set_ylabel("probabilite transmise")
    ax2.set_title("Montee de la probabilite transmise vers son plateau T")
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  PARTIE C : influence de la LARGEUR a de la barriere (V0 fixe).
# ---------------------------------------------------------------------------
def partie_C_influence_largeur():
    x, dt = grille()
    V0 = 4.0
    print("=== Partie C : influence de la largeur a (V0 =", V0, ") ===")

    x_det = 7.0   # detecteur FIXE (independant de a) pour comparer les temps
    largeurs = [0.5, 1.0, 1.5, 2.0, 2.5]
    T_num_liste = []
    T_th_liste = []
    retard_liste = []

    # reference : arrivee du paquet LIBRE au detecteur (calculee une fois)
    V_libre = construire_barriere(x, DEBUT_BARRIERE, 1.0, 0.0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    t_l, norme_l, xmoy_l, ptrans_l, xtrans_l, img_l, ti_l = simuler(
        x, V_libre, psi0, dt, NT, DEBUT_BARRIERE, nb_images=2)
    t_arr_libre = temps_de_passage(t_l, xmoy_l, x_det)

    for largeur in largeurs:
        fin = DEBUT_BARRIERE + largeur
        V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
        psi0b = construire_paquet(x, X0, A_PAQUET, K0)
        t_b, norme_b, xmoy_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
            x, V, psi0b, dt, NT, fin, nb_images=2)
        T_num = ptrans_b[NT - 1]
        T_th = transmission_analytique(E0, V0, largeur)
        t_arr_trans = temps_de_passage(t_b, xtrans_b, x_det)
        if (t_arr_trans is not None) and (t_arr_libre is not None):
            retard = t_arr_trans - t_arr_libre
        else:
            retard = float("nan")
        T_num_liste.append(T_num)
        T_th_liste.append(T_th)
        retard_liste.append(retard)
        print("  a =", largeur, " T_num =", round(T_num, 4),
              " T_th =", round(T_th, 4), " retard =", round(retard, 3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(largeurs, T_num_liste, "ro-", label="T numerique (paquet)")
    ax1.semilogy(largeurs, T_th_liste, "k.--", label="T analytique (onde plane k0)")
    ax1.set_xlabel("largeur a de la barriere"); ax1.set_ylabel("transmission T")
    ax1.set_title("La transmission chute (~ exponentiellement) avec a")
    ax1.legend(); ax1.grid(True, which="both")

    ax2.plot(largeurs, retard_liste, "bs-")
    ax2.set_xlabel("largeur a de la barriere")
    ax2.set_ylabel("retard du paquet transmis (t_trans - t_libre)")
    ax2.set_title("Le temps tunnel ne croit pas comme a/v (filtrage + Hartman)")
    ax2.grid(True)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  PARTIE D : influence de la HAUTEUR V0 (largeur fixe). Inclut le cas E > V0.
# ---------------------------------------------------------------------------
def partie_D_influence_hauteur():
    x, dt = grille()
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur
    print("=== Partie D : influence de la hauteur V0 (a =", largeur,
          ", E0 =", E0, ") ===")

    hauteurs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    T_num_liste = []
    T_th_liste = []

    for V0 in hauteurs:
        V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
        psi0b = construire_paquet(x, X0, A_PAQUET, K0)
        t_b, norme_b, xmoy_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
            x, V, psi0b, dt, NT, fin, nb_images=2)
        T_num = ptrans_b[NT - 1]
        T_th = transmission_analytique(E0, V0, largeur)
        T_num_liste.append(T_num)
        T_th_liste.append(T_th)
        if E0 > V0:
            regime = "E>V0 (classiquement passe)"
        elif E0 < V0:
            regime = "E<V0 (effet tunnel)"
        else:
            regime = "E=V0 (cas limite, T=1/2)"
        print("  V0 =", V0, " T_num =", round(T_num, 4),
              " T_th =", round(T_th, 4), " ->", regime)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hauteurs, T_num_liste, "ro-", label="T numerique")
    ax.plot(hauteurs, T_th_liste, "k.--", label="T analytique (k0)")
    ax.axvline(E0, color="blue", linestyle=":", label="V0 = E0 (frontiere)")
    ax.set_xlabel("hauteur V0 de la barriere"); ax.set_ylabel("transmission T")
    ax.set_title("Transmission vs hauteur : a gauche E>V0, a droite effet tunnel")
    ax.legend(); ax.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
#  PARTIE E : comparaison avec le cas CLASSIQUE.
# ---------------------------------------------------------------------------
def partie_E_comparaison_classique():
    largeur = 1.0
    print("=== Partie E : comparaison quantique / classique ===")
    print("  Energie du paquet : E0 =", E0)
    for V0 in [1.0, 4.0]:
        T_th = transmission_analytique(E0, V0, largeur)
        print("  --- V0 =", V0, "---")
        if E0 > V0:
            print("    Classiquement : E0 > V0 -> la particule PASSE toujours (T_clas = 1).")
            print("    Quantiquement : T =", round(T_th, 4),
                  "(< 1 : il existe une probabilite de REFLEXION).")
        else:
            print("    Classiquement : E0 < V0 -> la particule est TOUJOURS REFLECHIE (T_clas = 0).")
            print("    Quantiquement : T =", round(T_th, 4),
                  "(> 0 : EFFET TUNNEL, la particule peut traverser).")


def main():
    # Menu procedural : "A", "B", "C", "D", "E" ou "tout".
    # ATTENTION : "tout" enchaine plusieurs simulations -> compter ~1 a 2 min.
    choix = "tout"

    if choix == "A":
        partie_A_visualisation()
    elif choix == "B":
        partie_B_temps()
    elif choix == "C":
        partie_C_influence_largeur()
    elif choix == "D":
        partie_D_influence_hauteur()
    elif choix == "E":
        partie_E_comparaison_classique()
    elif choix == "tout":
        partie_A_visualisation()
        partie_B_temps()
        partie_C_influence_largeur()
        partie_D_influence_hauteur()
        partie_E_comparaison_classique()
    else:
        print("Choix inconnu :", choix)


if __name__ == "__main__":
    main()
