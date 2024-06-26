"""Test basic setup of NEML2 models"""

import unittest
import os.path

import torch

torch.set_default_dtype(torch.double)

# Selectively enable
try:
    from pyzag.neml2 import model
    import neml2
except ImportError:
    raise unittest.SkipTest("NEML 2 not installed")


class TestValidModel(unittest.TestCase):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "correct_model.i"), "implicit_rate"
        )
        self.pmodel = model.NEML2Model(self.nmodel)

    def test_state_names(self):
        self.assertEqual(self.pmodel.state_names, [["S"], ["internal", "ep"]])

    def test_forces_names(self):
        self.assertEqual(self.pmodel.force_names, [["E"], ["t"]])

    def test_old_forces_names(self):
        self.assertEqual(self.pmodel.old_force_names, [["E"], ["t"]])

    def test_parameter_names(self):
        self.assertEqual(
            set([n for n, _ in self.pmodel.named_parameters()]),
            set(
                [
                    "elasticity_E",
                    "elasticity_nu",
                    "flow_rate_eta",
                    "flow_rate_n",
                    "isoharden_K",
                    "yield_sy",
                ]
            ),
        )

        self.assertEqual(
            set([n for n, _ in self.pmodel.named_parameters()]),
            set([n.replace(".", "_") for n in self.nmodel.named_parameters().keys()]),
        )


class TestParameterUpdateHasEffect(unittest.TestCase):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "correct_model.i"), "implicit_rate"
        )
        self.pmodel = model.NEML2Model(self.nmodel)

    def test_parameter_update(self):
        self.assertTrue(
            torch.allclose(
                self.pmodel.isoharden_K,
                self.nmodel.named_parameters()["isoharden.K"].tensor().tensor(),
            )
        )
        self.assertTrue(
            torch.allclose(
                self.pmodel.isoharden_K,
                torch.tensor(1000.0),
            )
        )

        self.pmodel.isoharden_K.data = torch.tensor(50.0)
        self.pmodel._update_parameter_values()
        self.assertTrue(
            torch.allclose(
                self.pmodel.isoharden_K,
                self.nmodel.named_parameters()["isoharden.K"].tensor().tensor(),
            )
        )
