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

"""Test dense block-operator algebra."""

from __future__ import annotations

import unittest

import torch

from pyzag.chunktime import (
    BidiagonalForwardOperator,
    BidiagonalThomasFactorization,
)
from pyzag.operators.dense import (
    DenseBlockJacobian,
    DenseBlockOperator,
    DenseBlockVector,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class _DensePackedOperatorTestMixin:
    """Construct an operator via ``_make_op``; subclasses choose lazy
    (``DenseBlockOperator``) or eager (``DenseBlockOperator.factored``)."""

    @staticmethod
    def _make_op(data):
        raise NotImplementedError

    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6

        A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        B = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)

        I = torch.eye(self.sblk, dtype=A.dtype).reshape(1, 1, self.sblk, self.sblk)
        self.A_data = A + 2.0 * I
        self.B_data = B + 2.0 * I

        self.x_data = torch.rand(self.nblk, self.sbat, self.sblk)
        self.x = DenseBlockVector(self.x_data)

        self.A = self._make_op(self.A_data)
        self.B = self._make_op(self.B_data)

    def test_properties(self):
        self.assertEqual(self.A.device, self.A_data.device)
        self.assertEqual(self.A.dtype, self.A_data.dtype)
        self.assertEqual(self.A.nblk, self.nblk)
        self.assertEqual(self.A.batch_size, self.sbat)
        self.assertEqual(len(self.A), self.nblk)

    def test_matvec(self):
        expected = torch.matmul(self.A_data, self.x_data.unsqueeze(-1)).squeeze(-1)
        result = self.A.matvec(self.x)
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, expected))

    def test_t_matvec(self):
        expected = torch.matmul(
            self.A_data.transpose(-1, -2), self.x_data.unsqueeze(-1)
        ).squeeze(-1)
        result = self.A.t_matvec(self.x)
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, expected))

    def test_solve(self):
        b_data = torch.matmul(self.A_data, self.x_data.unsqueeze(-1)).squeeze(-1)
        b = DenseBlockVector(b_data)
        result = self.A.solve(b)
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, self.x_data))

    def test_clone(self):
        C = self.A.clone()
        self.assertTrue(torch.allclose(C.data, self.A.data))
        self.assertIsNot(C, self.A)
        self.assertIsNot(C.data, self.A.data)

    def test_getitem_single(self):
        blk = self.A[3:4]
        self.assertEqual(blk.nblk, 1)
        self.assertEqual(blk.batch_size, self.sbat)
        self.assertTrue(torch.allclose(blk.data, self.A.data[3:4]))

    def test_getitem_window(self):
        win = self.A[2:5]
        self.assertEqual(win.nblk, 3)
        self.assertEqual(win.batch_size, self.sbat)
        self.assertTrue(torch.allclose(win.data, self.A.data[2:5]))

    def test_getitem_int(self):
        blk = self.A[3]
        self.assertEqual(blk.nblk, 1)
        self.assertTrue(torch.allclose(blk.data, self.A.data[3:4]))


class TestDenseBlockOperator(_DensePackedOperatorTestMixin, unittest.TestCase):
    """Default lazy LU: factorization deferred until first solve."""

    @staticmethod
    def _make_op(data):
        return DenseBlockOperator(data)


class TestDenseBlockOperatorFactored(_DensePackedOperatorTestMixin, unittest.TestCase):
    """Eager LU via ``DenseBlockOperator.factored(data)``."""

    @staticmethod
    def _make_op(data):
        return DenseBlockOperator.factored(data)


class TestDenseBlockOperatorViews(unittest.TestCase):
    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.data = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        self.op = DenseBlockOperator(self.data)

    def test_pad_front(self):
        out = self.op.pad_front(2)
        self.assertEqual(out.nblk, self.nblk + 2)
        self.assertTrue(torch.allclose(out.data[2:], self.data))
        self.assertTrue(torch.allclose(out.data[:2], torch.zeros_like(out.data[:2])))

    def test_slice_off_front(self):
        out = self.op[2:]
        self.assertEqual(out.nblk, self.nblk - 2)
        self.assertTrue(torch.allclose(out.data, self.data[2:]))

    def test_setitem_window(self):
        repl_data = torch.rand(3, self.sbat, self.sblk, self.sblk)
        repl = DenseBlockOperator(repl_data)

        out = self.op.clone()
        out[2:5] = repl

        expected = self.data.clone()
        expected[2:5] = repl_data
        self.assertTrue(torch.allclose(out.data, expected))


class TestDenseBlockOperatorPCRPrimitives(unittest.TestCase):
    """Verify pcr_init / pcr_reduce_level / pcr_finalize produce the same
    reduced system that the old monolithic reduce_block did."""

    def setUp(self):
        self.nblk = 8
        self.sbat = 5
        self.sblk = 6

        A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        B = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        rhs_data = torch.rand(self.nblk, self.sbat, self.sblk)

        I = torch.eye(self.sblk, dtype=A.dtype).reshape(1, 1, self.sblk, self.sblk)
        self.A_data = A + 2.0 * I
        self.B_data = B
        self.rhs = DenseBlockVector(rhs_data)

        self.A = DenseBlockOperator.factored(self.A_data)
        self.B = DenseBlockOperator(self.B_data)

    def _run_pcr_loop(self, A_blk, B_blk, v_blk):
        """Run the full PCR loop via the new primitives and return (B_red, v_red)."""
        niter = A_blk.nblk.bit_length() - 1
        state = A_blk.pcr_init(B_blk, v_blk)
        for level in range(niter):
            state = A_blk.pcr_reduce_level(state, level)
        return A_blk.pcr_finalize(state)

    def test_output_types(self):
        B_red, v_red = self._run_pcr_loop(self.A, self.B, self.rhs)
        self.assertIsInstance(B_red, DenseBlockOperator)
        self.assertIsInstance(v_red, DenseBlockVector)

    def test_output_shapes(self):
        B_red, v_red = self._run_pcr_loop(self.A, self.B, self.rhs)
        self.assertEqual(B_red.nblk, self.nblk - 1)
        self.assertEqual(v_red.nblk, self.nblk - 1)

    def test_pcr_solves_bidiagonal_system(self):
        """Full-solve via PCR primitives matches dense LU reference."""
        from pyzag.chunktime import BidiagonalPCRFactorization

        solver = BidiagonalPCRFactorization(self.A, self.B[: self.nblk - 1])
        result = solver.matvec(self.rhs)

        # Build the dense bidiagonal matrix and solve directly.
        n = self.nblk * self.sblk
        M = torch.zeros(self.sbat, n, n, dtype=self.A_data.dtype)
        for k in range(self.nblk):
            r = k * self.sblk
            M[:, r : r + self.sblk, r : r + self.sblk] = self.A_data[k]
        for k in range(1, self.nblk):
            r = k * self.sblk
            M[:, r : r + self.sblk, (r - self.sblk) : r] = self.B_data[k - 1]
        rhs_flat = self.rhs.data.permute(1, 0, 2).reshape(self.sbat, -1).unsqueeze(-1)
        expected_flat = torch.linalg.solve(M, rhs_flat).squeeze(-1)
        expected = expected_flat.reshape(self.sbat, self.nblk, self.sblk).permute(
            1, 0, 2
        )

        self.assertTrue(torch.allclose(result.data, expected, atol=1e-10))

    def test_no_reduce_block_attribute(self):
        self.assertFalse(hasattr(self.A, "reduce_block"))


class TestDenseBlockJacobian(unittest.TestCase):
    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        I = torch.eye(self.sblk, dtype=torch.float64).reshape(
            1, 1, self.sblk, self.sblk
        )
        self.diag = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk) + 2.0 * I
        self.sub = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk) + 2.0 * I
        self.J_stacked = torch.stack([self.sub, self.diag])
        self.J = DenseBlockJacobian(self.diag, self.sub)

    def test_properties(self):
        self.assertEqual(self.J.nblk_steps, self.nblk)
        self.assertEqual(self.J.batch_size, self.sbat)
        self.assertEqual(self.J.block_size, self.sblk)
        self.assertEqual(self.J.dtype, self.diag.dtype)
        self.assertEqual(self.J.device, self.diag.device)

    def test_construction_validation(self):
        with self.assertRaises(ValueError):
            DenseBlockJacobian(self.diag, self.sub[..., :-1])  # mismatched shapes
        with self.assertRaises(ValueError):
            DenseBlockJacobian(self.diag[0], self.sub[0])  # wrong ndim

    def test_from_stacked(self):
        J_alt = DenseBlockJacobian.from_stacked(self.J_stacked)
        self.assertTrue(torch.allclose(J_alt.diag, self.diag))
        self.assertTrue(torch.allclose(J_alt.sub, self.sub))

    def test_from_stacked_validation(self):
        with self.assertRaises(ValueError):
            DenseBlockJacobian.from_stacked(self.diag)  # wrong ndim
        with self.assertRaises(ValueError):
            DenseBlockJacobian.from_stacked(self.J_stacked[:1])  # wrong leading dim

    def test_forward_system(self):
        sys = self.J.forward_system(BidiagonalThomasFactorization)
        self.assertIsInstance(sys, BidiagonalForwardOperator)
        self.assertEqual(sys.A.nblk, self.nblk)
        self.assertEqual(sys.B.nblk, self.nblk - 1)
        self.assertTrue(torch.allclose(sys.A.data, self.diag))
        self.assertTrue(torch.allclose(sys.B.data, self.sub[1:]))

    def test_forward_system_rejects_walk_order(self):
        walk = self.J.as_adjoint_walk()
        with self.assertRaises(RuntimeError):
            walk.forward_system(BidiagonalThomasFactorization)

    def test_adjoint_system(self):
        walk = self.J.as_adjoint_walk()
        sys = walk.adjoint_system(BidiagonalThomasFactorization)
        # adjoint_system returns the SOLVE operator directly (not a
        # BidiagonalForwardOperator -- different from forward_system).
        self.assertIsInstance(sys, BidiagonalThomasFactorization)
        self.assertEqual(sys.A.nblk, self.nblk - 1)
        self.assertEqual(sys.B.nblk, self.nblk - 2)
        # Reproduce the legacy `J.flip(1); A = J[1, 1:].T; B = J[0, 1:-1].T`
        flipped_diag = self.diag.flip(0)
        flipped_sub = self.sub.flip(0)
        self.assertTrue(torch.allclose(sys.A.data, flipped_diag[1:].transpose(-1, -2)))
        self.assertTrue(torch.allclose(sys.B.data, flipped_sub[1:-1].transpose(-1, -2)))

    def test_adjoint_system_rejects_forward_order(self):
        with self.assertRaises(RuntimeError):
            self.J.adjoint_system(BidiagonalThomasFactorization)

    def test_solve_terminal_adjoint(self):
        g = torch.rand(self.sbat, self.sblk)
        out = self.J.solve_terminal_adjoint(g)
        self.assertIsInstance(out, DenseBlockVector)
        self.assertEqual(out.nblk, 1)
        # Reproduce the legacy `-solve(J[1, -1].T, g)` operation
        expected = -torch.linalg.solve(self.diag[-1].transpose(-1, -2), g)
        self.assertTrue(torch.allclose(out.data[0], expected))

    def test_solve_terminal_adjoint_independent_of_walk(self):
        g = torch.rand(self.sbat, self.sblk)
        forward = self.J.solve_terminal_adjoint(g)
        walk = self.J.as_adjoint_walk().solve_terminal_adjoint(g)
        self.assertTrue(torch.allclose(forward.data, walk.data))

    def test_couple_prev_chunk(self):
        a_prev = DenseBlockVector(torch.rand(1, self.sbat, self.sblk))
        walk = self.J.as_adjoint_walk()
        out = walk.couple_prev_chunk(a_prev)
        self.assertIsInstance(out, DenseBlockVector)
        self.assertEqual(out.nblk, 1)
        # Reproduce the legacy `J.flip(1); J[0, 0].T @ a_prev[0]` operation
        expected = torch.matmul(
            self.sub.flip(0)[0].transpose(-1, -2),
            a_prev.data[0].unsqueeze(-1),
        ).squeeze(-1)
        self.assertTrue(torch.allclose(out.data[0], expected))

    def test_as_adjoint_walk_idempotent(self):
        walk = self.J.as_adjoint_walk()
        unwalk = walk.as_adjoint_walk()
        self.assertFalse(unwalk._reversed)
        self.assertTrue(torch.allclose(unwalk.diag, self.diag))
        self.assertTrue(torch.allclose(unwalk.sub, self.sub))

    def test_as_adjoint_walk_shares_storage(self):
        walk = self.J.as_adjoint_walk()
        # Backend should be lazy: no copy of underlying tensors
        self.assertIs(walk.diag, self.diag)
        self.assertIs(walk.sub, self.sub)


class TestDenseBlockOperatorFactories(unittest.TestCase):
    def test_identity(self):
        nblk, sbat, sblk = 4, 3, 5
        I_op = DenseBlockOperator.identity(
            nblk, sbat, sblk, torch.float64, torch.device("cpu")
        )
        self.assertEqual(I_op.nblk, nblk)
        self.assertEqual(I_op.batch_size, sbat)

        x = DenseBlockVector(torch.rand(nblk, sbat, sblk))
        result = I_op.matvec(x)
        self.assertTrue(torch.allclose(result.data, x.data))

    def test_from_diagonal(self):
        data = torch.rand(4, 3, 5, 5)
        op = DenseBlockOperator.from_diagonal(data)
        self.assertIsInstance(op, DenseBlockOperator)
        self.assertTrue(torch.allclose(op.data, data))


if __name__ == "__main__":
    unittest.main()
