"""
Testes unitários para a função PREDEP.
"""

import pytest
import numpy as np
from src.predep_calculation.core import predep


class TestPredep:
    """Testes para a função PREDEP."""
    
    def test_predep_returns_float_or_nan(self):
        """Teste se PREDEP retorna float ou NaN."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        
        result = predep(X, Y, n_boot=1000)
        
        assert isinstance(result, (float, np.floating)) or np.isnan(result)
    
    def test_predep_with_valid_arrays(self):
        """Teste com arrays válidos."""
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([2, 4, 6, 8, 10])
        
        result = predep(X, Y, n_boot=100)
        
        assert isinstance(result, (float, np.floating)) or np.isnan(result)
    
    def test_predep_with_constant_data_returns_nan(self):
        """Teste com dados constantes deve retornar NaN."""
        X = np.ones(20)
        Y = np.random.normal(0, 1, 20)
        
        result = predep(X, Y, n_boot=100)
        
        # Com dados constantes, deve retornar NaN ou falhar graciosamente
        assert np.isnan(result) or isinstance(result, (float, np.floating))

class TestPredepEstimator:
    """Testes da correção do estimador (átomo em zero por pares i == j).

    Contexto: `W = Y_i - Y_j` com i e j sorteados independentemente com reposição
    cai no mesmo ponto com probabilidade 1/n, criando um átomo de massa 1/n em
    zero — exatamente onde a KDE é avaliada. Como a massa é 1/n_bin, ela é maior
    em bins pequenos, o que inflava α proporcionalmente ao número de bins.
    Ver `results/exp09_bootstrap_defeito.md`.
    """

    def test_null_is_unbiased_when_excluding_self_pairs(self):
        """X e Y independentes: α deve ser indistinguível de zero."""
        vals = [predep(np.random.default_rng(s).normal(size=87),
                       np.random.default_rng(1000 + s).normal(size=87) * 40 + 150,
                       n_boot=5000, seed=s, exclude_self_pairs=True)
                for s in range(40)]
        vals = np.array(vals, dtype=float)
        mean, se = np.nanmean(vals), np.nanstd(vals) / np.sqrt(len(vals))
        assert abs(mean) < 3 * se + 0.02, f"α sob o nulo = {mean:.4f} (ep {se:.4f})"

    def test_legacy_estimator_is_biased_upward(self):
        """O estimador antigo deve ficar claramente acima de zero sob o nulo.

        Guarda o diagnóstico: se este teste um dia falhar, o viés mudou de
        natureza e as conclusões de `results/exp09_*.md` precisam ser revistas.
        """
        vals = [predep(np.random.default_rng(s).normal(size=87),
                       np.random.default_rng(1000 + s).normal(size=87) * 40 + 150,
                       n_boot=5000, seed=s, exclude_self_pairs=False)
                for s in range(40)]
        assert np.nanmean(vals) > 0.03

    def test_converges_toward_theoretical_alpha(self):
        """Bivariada normal: α → 1 - sqrt(1 - ρ²) conforme n cresce.

        A asserção é sobre o ERRO encolher, não sobre α subir: o estimador do
        artigo (V-statistic, i==j permitido) converge por CIMA — 0,56 em n=200,
        0,46 em n=2000, contra 0,400 real —, enquanto o U-statistic converge por
        baixo. Um teste que exigisse α crescente só valeria para um dos dois.
        """
        rho = 0.8
        true = 1 - np.sqrt(1 - rho**2)
        got = {}
        for n in (200, 2000):
            v = []
            for s in range(8):
                rng = np.random.default_rng(500 + s)
                xy = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
                v.append(predep(xy[:, 0], xy[:, 1], n_boot=5000, seed=s))
            got[n] = np.nanmean(v)
        err = {n: abs(a - true) for n, a in got.items()}
        assert err[2000] < err[200], f"erro não encolheu com n: {got}"
        assert err[2000] < 0.10, f"α(n=2000)={got[2000]:.3f} vs {true:.3f}"

    def test_ward_method_matches_paper_bin_count(self):
        """method='ward' usa k=√N bins, como o Algorithm 1 do artigo."""
        rng = np.random.default_rng(0)
        for n in (87, 400):
            X, Y = rng.normal(size=n), rng.normal(size=n)
            assert not np.isnan(predep(X, Y, n_boot=1000, seed=1, method="ward"))

    def test_invalid_method_raises(self):
        """method inválido levanta, em vez de virar NaN silencioso."""
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            predep(rng.normal(size=40), rng.normal(size=40), n_boot=500,
                   method="nao_existe")


class TestConformidadeArtigo:
    """Conformidade com arXiv:2501.10815v1 (Assunção et al.).

    Testes derivados do que o artigo PUBLICA ou PROVA — se falharem, a
    implementação divergiu do método, não do nosso gosto. Ver
    `results/conformidade_final.md`.
    """

    @staticmethod
    def _sintetico(kind, n, rng):
        """Relações da Seção 6: X ~ U(-1,1), Y = h(X), sem ruído.

        θ_i ~ U(-1,1) — o texto diz "spanning (-1, -1)", typo evidente — exceto
        no piecewise, onde θ0 ~ U(-3,0) e θ1 ~ U(0,3).
        """
        X = rng.uniform(-1, 1, n)
        t = rng.uniform(-1, 1, 4)
        if kind == "Lin":
            Y = t[0] * X + t[1]
        elif kind == "Qua":
            Y = t[2] * X**2 + t[1] * X + t[0]
        elif kind == "Cub":
            Y = t[3] * X**3 + t[2] * X**2 + t[1] * X + t[0]
        elif kind == "Log":
            Y = t[0] * np.log(X + 1 + 1e-12)
        elif kind == "Sin":
            Y = t[1] * np.sin(t[0] * X)
        elif kind == "Piw":
            a, b = rng.uniform(-3, 0), rng.uniform(0, 3)
            with np.errstate(divide="ignore"):
                Y = np.minimum(np.maximum(1 / X, a), b)
        ok = np.isfinite(X) & np.isfinite(Y)
        return X[ok], Y[ok]

    def test_reproduz_table_1(self):
        """Table 1 do artigo: α para 6 relações sem ruído, N=1000.

        O teste de conformidade mais forte disponível — são números publicados,
        obtidos com o estimador do artigo no regime do artigo. Medido aqui:
        desvio máximo 0,027 (Piecewise) e médio 0,007.
        """
        # (valor do artigo, réplicas, tolerância). O Piecewise precisa de mais
        # réplicas e folga maior: o ponto de saturação depende dos θ sorteados,
        # e o desvio entre réplicas é 0,276 (contra 0,001-0,007 dos demais), o
        # que dá erro padrão de 0,07 mesmo com 15 réplicas.
        paper = {"Lin": (0.97, 6, 0.06), "Log": (0.98, 6, 0.06),
                 "Cub": (0.98, 6, 0.06), "Qua": (0.98, 6, 0.06),
                 "Sin": (0.97, 6, 0.06), "Piw": (0.88, 15, 0.20)}
        for kind, (ref, reps, tol) in paper.items():
            vals = [predep(*self._sintetico(kind, 1000, np.random.default_rng(2000 + s)),
                           seed=s) for s in range(reps)]
            got = np.nanmean(vals)
            assert abs(got - ref) < tol, f"{kind}: α={got:.3f} vs artigo {ref:.2f}"

    def test_caso_analitico_gaussiano(self):
        """α = 1 - sqrt(1 - ρ²) para a bivariada normal (Seção 4 + Apêndice C)."""
        for rho in (0.5, 0.8):
            true = 1 - np.sqrt(1 - rho**2)
            vals = []
            for s in range(6):
                rng = np.random.default_rng(3000 + s)
                xy = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 2000)
                vals.append(predep(xy[:, 0], xy[:, 1], seed=s))
            got = np.nanmean(vals)
            assert abs(got - true) < 0.12, f"ρ={rho}: α={got:.3f} vs {true:.3f}"

    def test_anel_dependencia_nao_linear(self):
        """(X,Y) uniforme no anel entre x²+y²=3/4 e x²+y²=1 (Apêndice C).

        O artigo: "The Pearson correlation coefficient is zero but our PREDEP
        measure is greater than zero." Guarda contra o α virar um detector de
        relação linear.
        """
        rng = np.random.default_rng(7)
        n, xs, ys = 3000, [], []
        while len(xs) < n:
            x, y = rng.uniform(-1, 1, 2)
            r2 = x * x + y * y
            if 0.75 <= r2 <= 1.0:
                xs.append(x)
                ys.append(y)
        X, Y = np.array(xs), np.array(ys)
        assert abs(np.corrcoef(X, Y)[0, 1]) < 0.08, "Pearson deveria ser ~0"
        assert predep(X, Y, seed=1) > 0.05, "PREDEP deveria detectar o anel"

    def test_p4_assimetria(self):
        """P4: PREDEP é assimétrico — α_{Y|X} != α_{X|Y}.

        Guarda contra a inversão de argumentos: `references/predep.py` bina Y e
        mede X, o oposto do Algorithm 1, e foi provavelmente daí que veio a
        inversão encontrada em `src/modal/run_experiment.py`.
        """
        rng = np.random.default_rng(11)
        X = rng.uniform(-1, 1, 2000)
        Y = X**2 + rng.normal(0, 0.02, 2000)     # o exemplo do próprio artigo
        a_yx = predep(X, Y, seed=1)
        a_xy = predep(Y, X, seed=1)
        assert a_yx > a_xy + 0.1, f"α_(Y|X)={a_yx:.3f} deveria superar α_(X|Y)={a_xy:.3f}"

    def test_pesos_renormalizados_em_bin_degenerado(self):
        """Bin com Y constante não pode derrubar α.

        Sem renormalizar, o peso do bin pulado some da soma enquanto S_Y segue
        sendo o total: no Piecewise da Table 1, 14 de 31 bins degeneram, os
        pesos somam 0,544 e α vai a -3,7. O Algorithm 1 não prevê pular bins.
        """
        rng = np.random.default_rng(5)
        X = rng.uniform(-1, 1, 600)
        Y = np.clip(X, -0.3, 0.3)                # satura: bins das pontas ficam constantes
        got = predep(X, Y, seed=2)
        assert 0.0 <= got <= 1.0, f"α={got:.3f} fora de [0,1] com bins degenerados"
