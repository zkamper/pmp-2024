from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dimensiunea gridului
dimensiune_grid = (10, 10)

# Lista de culori predefinite
culori = [
    "red", "blue", "green", "yellow", 
    "purple", "orange", "pink", "cyan", 
    "brown", "lime"
]

# Citirea gridului
df = pd.read_csv('grid_culori.csv',header=None)
grid_culori = df.to_numpy()
print(grid_culori)

# Generarea secvenței de culori observate
observatii = ['red', 'red', 'lime', 'yellow', 'blue']

# Mapare culori -> indecși
culoare_to_idx = {culoare: idx for idx, culoare in enumerate(culori)}
idx_to_culoare = {idx: culoare for culoare, idx in culoare_to_idx.items()}

# Transformăm secvența de observații în indecși
observatii_idx = [culoare_to_idx[c] for c in observatii]

# Definim stările ascunse ca fiind toate pozițiile din grid (100 de stări)
numar_stari = dimensiune_grid[0] * dimensiune_grid[1]
stari_ascunse = [(i, j) for i in range(dimensiune_grid[0]) for j in range(dimensiune_grid[1])]
stare_to_idx = {stare: idx for idx, stare in enumerate(stari_ascunse)}
idx_to_stare = {idx: stare for stare, idx in stare_to_idx.items()}

# Matrice de tranziție
transitions = np.zeros((numar_stari, numar_stari))
for i, j in stari_ascunse:
    vecini = [
        (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)  # sus, jos, stânga, dreapta
    ]
    vecini_valizi = [stare_to_idx[(x, y)] for x, y in vecini if 0 <= x < 10 and 0 <= y < 10]
    stari_ascunse = [idx_to_stare[i] for i in vecini_valizi]
    if (i,j-1) in stari_ascunse:
        # 1 - A

        # stare_cur = stare_to_idx[(i,j)]
        # stare_noua = stare_to_idx[(i,j-1)]
        # transitions[stare_cur][stare_noua] = 0.4
        # stari_ascunse.remove((i,j-1))
        # remaining_states = len(stari_ascunse)
        # for stare_ascunsa in stari_ascunse:
        #     stare_cur = stare_to_idx[(i, j)]
        #     stare_noua = stare_to_idx[stare_ascunsa]
        #     transitions[stare_cur][stare_noua] = 0.6 / remaining_states

        # 1 - B

        stari_ascunse.remove((i,j-1))
        remaining_states = len(stari_ascunse)
        for stare_ascunsa in stari_ascunse:
            stare_cur = stare_to_idx[(i, j)]
            stare_noua = stare_to_idx[stare_ascunsa]
            transitions[stare_cur][stare_noua] = 1 / remaining_states

        # EXPLICATIE
        # in cel mai probabil drum, agentul NU alege stanga niciodata, deci daca eliminam posibilitatea
        # de a face stanga, cel mai probabil drum nu o sa se schimbe

    else:
        remaining_states = len(stari_ascunse)
        for stare_ascunsa in stari_ascunse:
            stare_cur = stare_to_idx[(i, j)]
            stare_noua = stare_to_idx[(stare_ascunsa[0], stare_ascunsa[1])]
            transitions[stare_cur][stare_noua] = 1 / remaining_states


# Matrice de emisie
emissions = np.zeros((numar_stari, len(culori)))
for idx_stare in range(numar_stari):
    stare = idx_to_stare[idx_stare]
    culoare = grid_culori[stare[0]][stare[1]]
    idx_culoare = culoare_to_idx[culoare]
    emissions[idx_stare][idx_culoare] = 1

# Generare probabilitate start
possible_starts = []
start_prob = np.zeros(numar_stari)
for idx_stare in range(numar_stari):
    stare = idx_to_stare[idx_stare]
    culoare = grid_culori[stare[0]][stare[1]]
    idx_culoare = culoare_to_idx[culoare]
    if culoare_to_idx[observatii[0]] == idx_culoare:
        possible_starts.append(idx_stare)

prob_start = 1 / len(possible_starts)
for possible_start in possible_starts:
    start_prob[possible_start] = prob_start
# Modelul HMM
model = hmm.CategoricalHMM(n_components=numar_stari)
model.startprob_ = start_prob
model.emissionprob_ = emissions
model.transmat_ = transitions


# Rulăm algoritmul Viterbi pentru secvența de observații
observation_sequence = np.array(observatii_idx).reshape(-1,1)
secventa_stari = model.predict(observation_sequence)

# Convertim secvența de stări în poziții din grid
drum = [idx_to_stare[idx] for idx in secventa_stari]

# Vizualizăm drumul pe grid
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(dimensiune_grid[0]):
    for j in range(dimensiune_grid[1]):
        culoare = grid_culori[i, j]
        ax.add_patch(plt.Rectangle((j, dimensiune_grid[0] - i - 1), 1, 1, color=culoare))
        ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, culoare, 
                color="white", ha="center", va="center", fontsize=8, fontweight="bold")

# Evidențiem drumul rezultat
for idx, (i, j) in enumerate(drum):
    ax.add_patch(plt.Circle((j + 0.5, dimensiune_grid[0] - i - 0.5), 0.3, color="black", alpha=0.7))
    ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, str(idx + 1),
            color="white", ha="center", va="center", fontsize=10, fontweight="bold")

# Setări axă
ax.set_xlim(0, dimensiune_grid[1])
ax.set_ylim(0, dimensiune_grid[0])
ax.set_xticks(range(dimensiune_grid[1]))
ax.set_yticks(range(dimensiune_grid[0]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(visible=True, color="black", linewidth=0.5)
ax.set_aspect("equal")
plt.title("Drumul rezultat al stărilor ascunse", fontsize=14)
plt.show()