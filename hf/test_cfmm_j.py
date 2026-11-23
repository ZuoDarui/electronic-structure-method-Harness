import math
import unittest

import numpy as np
from pyscf import df, gto, scf
from pyscf.df import df_jk

from cfmm_j import (CFMMRIJ, CoulombDerivatives, MultipoleTranslation,
                    cartesian_indices, harmonic_to_cartesian)
from make_alkane_xyz import format_xyz


def _alkane(n, basis):
    return gto.M(atom="\n".join(format_xyz(n).splitlines()[2:]), basis=basis,
                 unit="Angstrom", verbose=0)


class CoulombTensorTest(unittest.TestCase):
    def test_taylor_series_of_inverse_distance(self):
        order = 10
        index = cartesian_indices(order)
        r = np.array([3.1, -1.7, 2.2])
        s = np.array([0.3, 0.2, -0.25])
        fact = np.array([math.factorial(a) * math.factorial(b) * math.factorial(c)
                         for a, b, c in index])
        series = (np.prod(s ** index, axis=1) / fact * CoulombDerivatives(order)(r)[0]).sum()
        self.assertAlmostEqual(series, 1.0 / np.linalg.norm(r + s), delta=1e-11)

    def test_point_charge_interaction(self):
        order = 12
        index = cartesian_indices(order)
        rng = np.random.RandomState(3)
        qa, pa = rng.randn(4), rng.randn(4, 3) * 0.5
        qb, pb = rng.randn(3), rng.randn(3, 3) * 0.5
        shift = np.array([12.0, -3.0, 4.0])
        ma = (qa[:, None] * np.prod(pa[:, None] ** index[None], axis=2)).sum(0)
        mb = (qb[:, None] * np.prod(pb[:, None] ** index[None], axis=2)).sum(0)
        m2l = MultipoleTranslation(order)
        local = m2l.local((ma * m2l.source_scale)[None],
                          CoulombDerivatives(order)(shift)[None])[0]
        energy = (mb * m2l.target_scale * local).sum()
        exact = sum(qa[i] * qb[j] / np.linalg.norm(shift + pb[j] - pa[i])
                    for i in range(4) for j in range(3))
        self.assertAlmostEqual(energy, exact, delta=1e-12)

    def test_harmonic_map_shape(self):
        g = harmonic_to_cartesian(6)
        self.assertEqual(g.shape, (49, 84))


class CFMMRIJTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = _alkane(14, "cc-pvdz")
        cls.dm = scf.hf.init_guess_by_minao(cls.mol)
        dfobj = df.DF(cls.mol, auxbasis="weigend")
        dfobj.verbose = 0
        cls.ref = df_jk.get_j(dfobj, cls.dm, hermi=1)

    def test_matches_pyscf_ri_j(self):
        engine = CFMMRIJ(self.mol, order=12, theta=0.3, box_bohr=4.0)
        vj = engine.build(self.dm)
        self.assertGreater(engine.stats["far_interactions"], 0)
        self.assertLess(np.abs(vj - self.ref).max(), 1e-8)
        self.assertLess(abs(np.einsum("ij,ij", vj - self.ref, self.dm)), 1e-7)
        self.assertLess(np.abs(vj - vj.T).max(), 1e-12)

    def test_linear_in_density(self):
        engine = CFMMRIJ(self.mol, order=8, theta=0.4, box_bohr=4.0)
        rng = np.random.RandomState(0)
        delta = rng.randn(*self.dm.shape) * 1e-3
        delta = delta + delta.T
        full = engine.build(self.dm + delta)
        parts = engine.build(self.dm) + engine.build(delta)
        self.assertLess(np.abs(full - parts).max(), 1e-11)

    def test_rejects_unsupported_order(self):
        with self.assertRaises(ValueError):
            CFMMRIJ(self.mol, order=13)


if __name__ == "__main__":
    unittest.main()
