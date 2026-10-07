"""Treino do SOM de consenso no Modal: um container por estação.

Chamado por `tools/som_consenso_local.py --modal`. Cada container recebe a matriz
já padronizada de uma estação e treina as sementes em paralelo nos seus núcleos;
devolve `[(regimes, erro_bmu), ...]` na ordem das sementes. O voto entre sementes
e as assinaturas ficam no lado local, que é barato.
"""
import modal

image = (
    modal.Image.debian_slim()
    .pip_install(["pandas", "xarray[complete]", "numpy<2", "scipy", "astropy", "netcdf4",
                  "h5netcdf", "pyshp", "matplotlib", "minisom", "scikit-learn"])
    .add_local_python_source("src")
)
app = modal.App("predep-som-consenso", image=image)
NUCLEOS = 10


def _uma(args):
    from src.analysis.som_regimes import treinar_e_projetar

    return treinar_e_projetar(*args)


@app.function(cpu=NUCLEOS, memory=NUCLEOS * 1536, timeout=6 * 3600)
def treinar_estacao(xm, xa, n_regimes, sementes, iters):
    import os
    from multiprocessing import Pool

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with Pool(min(NUCLEOS, len(sementes))) as pool:
        saidas = pool.map(_uma, [(xm, xa, n_regimes, sd, iters) for sd in sementes])
    # listas puras: o container tem NumPy 2 e o ambiente local NumPy 1 — arrays
    # devolvidos via pickle não desserializam do lado local (numpy._core)
    return [(r.tolist(), e.tolist()) for r, e in saidas]


# ── sensibilidade a K: um SOM por semente, k-means dos neurônios para cada K ──
def _semente_varios_k(args):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    from src.analysis.som_insights import train_som

    xm, xa, ks, seed, iters = args
    som, _ = train_som(xm, seed=seed, num_iteration=iters, use_epochs=True)
    w = som.get_weights().reshape(-1, xa.shape[1])
    bmu = np.empty(len(xa), dtype=np.int32)
    for i in range(0, len(xa), 2000):
        b = xa[i:i + 2000]
        bmu[i:i + 2000] = np.linalg.norm(b[:, None, :] - w[None], axis=2).argmin(axis=1)
    rot, met = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(w)
        rot[k] = km.labels_[bmu].astype(np.int8).tobytes()
        met[k] = (float(silhouette_score(w, km.labels_)),
                  float(davies_bouldin_score(w, km.labels_)))
    return rot, met


@app.function(cpu=NUCLEOS, memory=NUCLEOS * 2048, timeout=8 * 3600)
def varrer_k_estacao(xm, xa, ks, sementes, iters):
    """Para cada semente: (rótulos por K em bytes int8, (silhouette, DB) por K)."""
    import os
    from multiprocessing import Pool

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with Pool(min(NUCLEOS, len(sementes))) as pool:
        return pool.map(_semente_varios_k,
                        [(xm, xa, list(ks), sd, iters) for sd in sementes])
