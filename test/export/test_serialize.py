# Owner(s): ["module: dynamo"]
import pickle
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import dynamic_dim, export
from torch._export.graph_module import get_export_meta
from torch._export.serialize import convert_fake_tensor_to_tensor_meta, convert_tensor_meta_to_fake_tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.experimental import control_flow


class TestSerialize(TestCase):
    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
    def test_pickle(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            def true_fn(x):
                def inner_true_fn(y):
                    return x + y

                return inner_true_fn(x)

            def false_fn(x):
                def inner_false_fn(y):
                    return x - y

                return inner_false_fn(x)

            return control_flow.cond(x.shape[0] < 10, true_fn, false_fn, [x])

        inputs = (torch.ones(3),)
        mmep = export(f, inputs)
        gm = mmep.find_method("forward")
        gm.print_readable()

        # Pickle the ExportGraphModule
        pickled_gm = pickle.dumps(convert_fake_tensor_to_tensor_meta(gm)[0])
        loaded_gm = convert_tensor_meta_to_fake_tensor(pickle.loads(pickled_gm))

        for node1, node2 in zip(loaded_gm.graph.nodes, gm.graph.nodes):
            val1 = node1.meta.get("val", None)
            val2 = node2.meta.get("val", None)

            if val1 is None or val2 is None:
                # Either both are None
                self.assertEqual(val1, val2)
            elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                # Or both are fake tensors with the same shape/dtype
                self.assertEqual(val1.shape, val2.shape)
                self.assertEqual(val1.dtype, val2.dtype)
            elif isinstance(val1, list) and isinstance(val2, list):
                # Or both are fake tensors lists with one element and with the
                # same shape/dtype
                self.assertTrue(len(val1) == len(val2) and len(val1) == 1)
                self.assertEqual(val1[0].shape, val2[0].shape)
                self.assertEqual(val1[0].dtype, val2[0].dtype)
            else:
                # For expressions like 's0 < 10' can only compare through string
                self.assertEqual(str(val1), str(val2))

        self.assertTrue(torch.allclose(loaded_gm(*inputs), gm(*inputs)))

        # Check metadata
        orig_meta = get_export_meta(gm)
        new_meta = get_export_meta(loaded_gm)
        self.assertEqual(orig_meta.in_spec, new_meta.in_spec)
        self.assertEqual(orig_meta.out_spec, new_meta.out_spec)

    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
    def test_pickle_dynamic(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        constraints = [
            dynamic_dim(inputs[0], 0),
            dynamic_dim(inputs[2], 0),
            dynamic_dim(inputs[2], 0) == dynamic_dim(inputs[0], 0),
        ]
        mmep = export(DynamicShapeSimpleModel(), inputs, constraints)
        gm = mmep.find_method("forward")

        # Pickle the ExportGraphModule
        pickled_gm = pickle.dumps(convert_fake_tensor_to_tensor_meta(gm))
        loaded_gm = convert_tensor_meta_to_fake_tensor(*pickle.loads(pickled_gm))

        # Check fake tensor metadata
        shape_env1, shape_env2 = None, None
        for node1, node2 in zip(loaded_gm.graph.nodes, gm.graph.nodes):
            val1 = node1.meta.get("val", None)
            val2 = node2.meta.get("val", None)

            if val1 is None or val2 is None:
                # Either both are None
                self.assertEqual(val1, val2)
            elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                # Or both are fake tensors with the same shape/dtype
                self.assertEqual(val1.shape, val2.shape)
                self.assertEqual(val1.dtype, val2.dtype)

                if shape_env1 is None:
                    shape_env1 = val1.fake_mode.shape_env
                    shape_env2 = val2.fake_mode.shape_env
            elif isinstance(val1, list) and isinstance(val2, list):
                # Or both are fake tensors lists with one element and with the
                # same shape/dtype
                self.assertTrue(len(val1) == len(val2) and len(val1) == 1)
                self.assertEqual(val1[0].shape, val2[0].shape)
                self.assertEqual(val1[0].dtype, val2[0].dtype)

                if shape_env1 is None:
                    shape_env1 = val1[0].fake_mode.shape_env
                    shape_env2 = val2[0].fake_mode.shape_env
            else:
                self.assertEqual(val1, val2)

        # Check the shape env
        self.assertEqual(shape_env1.guards, shape_env2.guards)
        self.assertEqual(shape_env1.var_to_val, shape_env2.var_to_val)
        self.assertEqual(shape_env1.var_to_sources, shape_env2.var_to_sources)
        self.assertEqual(shape_env1.dim_constraints._univariate_inequalities, shape_env2.dim_constraints._univariate_inequalities)
        self.assertEqual(shape_env1.dim_constraints._dynamic_results, shape_env2.dim_constraints._dynamic_results)

        # Check correctness
        self.assertTrue(torch.allclose(loaded_gm(*inputs), gm(*inputs)))

        # Check metadata
        orig_meta = get_export_meta(gm)
        new_meta = get_export_meta(loaded_gm)
        self.assertEqual(orig_meta.in_spec, new_meta.in_spec)
        self.assertEqual(orig_meta.out_spec, new_meta.out_spec)

if __name__ == '__main__':
    run_tests()
