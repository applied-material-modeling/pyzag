# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: pyzag
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Test linear algebra on blocked diagonal matrices"""

from pyzag import chunktime

# fix test to account for new classes
from pyzag.operators.dense import DenseBlockLUFactorizedOperator, DenseBlockOperator

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
        self.blk_B = torch.rand(self.nblk - 1, self.sbat, self.sblk, self.sblk) / 10

        # Diagonal blocks must be solvable operators
        self.Aop = DenseBlockLUFactorizedOperator(self.blk_A)
        self.Bop = DenseBlockOperator(self.blk_B)

        self.A = chunktime.BidiagonalForwardOperator(self.Aop, self.Bop)
        self.b = torch.rand(self.nblk, self.sbat, self.sblk)

    def _transform_rhs(self, b):
        return b.transpose(0, 1).flatten(1)

    def _transform_soln(self, x):
        return x.reshape((self.sbat, self.nblk, self.sblk)).transpose(0, 1)

    def _dense_matrix(self):
        return chunktime.SquareBatchedBlockDiagonalMatrix(
            [self.blk_A, self.blk_B], [0, -1]
        ).to_dense()

    def test_inv_mat_vec_thomas(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalThomasFactorization(self.Aop, self.Bop)
            one = self._transform_soln(
                torch.linalg.solve(self._dense_matrix(), self._transform_rhs(self.b))
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_pcr(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalPCRFactorization(self.Aop, self.Bop)
            one = self._transform_soln(
                torch.linalg.solve(self._dense_matrix(), self._transform_rhs(self.b))
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_pcr(self):
        """Hybrid method, but set min_size so it always uses PCR"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(self.Aop, self.Bop)
            one = self._transform_soln(
                torch.linalg.solve(self._dense_matrix(), self._transform_rhs(self.b))
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_thomas(self):
        """Hybrid method, but set min_size so it always uses Thomas"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(
                self.Aop, self.Bop, min_size=self.max_nblk + 1
            )
            one = self._transform_soln(
                torch.linalg.solve(self._dense_matrix(), self._transform_rhs(self.b))
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_actual(self):
        """Hybrid method actually set to do something"""
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorizationImpl(
                self.Aop, self.Bop, min_size=self.nblk // 2
            )
            one = self._transform_soln(
                torch.linalg.solve(self._dense_matrix(), self._transform_rhs(self.b))
            )
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_mat_vec(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            one = self._transform_soln(
                self._dense_matrix()
                .matmul(self._transform_rhs(self.b).unsqueeze(-1))
                .squeeze(-1)
            )
            two = self.A(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_vec_mat(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            one = self._transform_soln(
                self._dense_matrix()
                .transpose(-1, -2)
                .matmul(self._transform_rhs(self.b).unsqueeze(-1))
                .squeeze(-1)
            )
            two = self.A.vecmat(self.b)

            self.assertTrue(torch.allclose(one, two))
