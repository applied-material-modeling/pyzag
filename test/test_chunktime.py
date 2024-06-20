"""Test linear algebra on blocked diagonal matrices"""

from pyzag import chunktime

import torch

import unittest

torch.set_default_dtype(torch.float64)


class TestBackwardEulerChunkTimeOperator(unittest.TestCase):
    def setUp(self):
        self.sblk = 6
        self.max_nblk = 31
        self.sbat = 5

    def _gen_operators(self):
        self.blk_A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        self.blk_B = (
            torch.rand(self.nblk - 1, self.sbat, self.sblk, self.sblk) / 10
        )  # Diagonal dominance

        self.A = chunktime.BidiagonalForwardOperator(self.blk_A, self.blk_B)
        self.b = torch.rand(self.nblk, self.sbat, self.sblk)

    def _transform_rhs(self, b):
        return b.transpose(0, 1).flatten(1)

    def _transform_soln(self, x):
        return x.reshape((self.sbat, self.nblk, self.sblk)).transpose(0, 1)

    def test_inv_mat_vec_thomas(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalThomasFactorization(self.blk_A, self.blk_B)
            one = self._transform_soln(
                torch.linalg.solve(
                    self.A.to_diag().to_dense(), self._transform_rhs(self.b)
                )
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_pcr(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalPCRFactorization(self.blk_A, self.blk_B)
            one = self._transform_soln(
                torch.linalg.solve(
                    self.A.to_diag().to_dense(), self._transform_rhs(self.b)
                )
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_pcr(self):
        """Hybrid method, but set min_size so it always uses PCR"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(self.blk_A, self.blk_B)
            one = self._transform_soln(
                torch.linalg.solve(
                    self.A.to_diag().to_dense(), self._transform_rhs(self.b)
                )
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_thomas(self):
        """Hybrid method, but set min_size so it always uses Thomas"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(
                self.blk_A, self.blk_B, min_size=self.max_nblk + 1
            )
            one = self._transform_soln(
                torch.linalg.solve(
                    self.A.to_diag().to_dense(), self._transform_rhs(self.b)
                )
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_actual(self):
        """Hybrid method actually set to do something"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(
                self.blk_A, self.blk_B, min_size=self.nblk // 2
            )
            one = self._transform_soln(
                torch.linalg.solve(
                    self.A.to_diag().to_dense(), self._transform_rhs(self.b)
                )
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_mat_vec(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            one = self._transform_soln(
                self.A.to_diag()
                .to_dense()
                .matmul(self._transform_rhs(self.b).unsqueeze(-1))
                .squeeze(-1)
            )
            two = self.A(self.b)

            self.assertTrue(torch.allclose(one, two))
