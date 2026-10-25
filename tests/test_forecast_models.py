"""Synthetic tests for shared/forecast_models.py. Small CPU tensors only; no real data."""
import io
import math
import unittest

import numpy as np
import torch

from shared import forecast_models as fm


def batch(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 72, 1, generator=g)


def make(kind, seed=0):
    torch.manual_seed(seed)
    return fm.build_model(kind).eval()


class ContractTests(unittest.TestCase):
    def test_shapes_and_return_contract(self):
        x = batch()
        out, attn = make("lstm")(x)
        self.assertEqual(out.shape, (5, 24))
        self.assertIsNone(attn)
        out, attn = make("attention")(x)
        self.assertEqual(out.shape, (5, 24))
        self.assertEqual(attn.shape, (5, 72, 72))
        self.assertEqual(out.dtype, torch.float32)

    def test_finite_outputs(self):
        for kind in fm.MODEL_TYPES:
            out, attn = make(kind)(batch())
            self.assertTrue(torch.isfinite(out).all())
            if attn is not None:
                self.assertTrue(torch.isfinite(attn).all())

    def test_input_validation(self):
        for kind in fm.MODEL_TYPES:
            m = make(kind)
            for bad in (torch.zeros(2, 71, 1), torch.zeros(2, 72), torch.zeros(2, 72, 2),
                        torch.zeros(2, 72, 1, dtype=torch.float64)):
                with self.assertRaises(ValueError):
                    m(bad)

    def test_unknown_model_type(self):
        with self.assertRaises(ValueError):
            fm.build_model("gru")

    def test_baseline_has_no_attention_parameters(self):
        names = [n for n, _ in make("lstm").named_parameters()]
        self.assertTrue(all(n.startswith(("lstm.", "head.")) for n in names), names)

    def test_parameter_counts_reported(self):
        a, b = fm.count_parameters(make("lstm")), fm.count_parameters(make("attention"))
        print(f"\nparameters: lstm={a:,} attention={b:,}")
        self.assertGreater(a, 0)
        self.assertGreater(b, 0)
        self.assertLess(max(a, b), 50_000)  # modest, local-CPU sized


class AttentionMathTests(unittest.TestCase):
    def test_rows_sum_to_one_and_nonnegative(self):
        _, attn = make("attention")(batch())
        self.assertTrue((attn >= 0).all())
        np.testing.assert_allclose(attn.sum(-1).detach().numpy(), 1.0, atol=1e-5)

    def test_sdpa_matches_numpy_reference(self):
        rng = np.random.default_rng(1)
        q, k, v = (rng.normal(size=(2, 7, 5)) for _ in range(3))
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(5)
        e = np.exp(scores - scores.max(-1, keepdims=True))
        w_ref = e / e.sum(-1, keepdims=True)
        out_ref = w_ref @ v
        out, w = fm.scaled_dot_product_attention(*(torch.tensor(a) for a in (q, k, v)))
        np.testing.assert_allclose(w.numpy(), w_ref, atol=1e-12)
        np.testing.assert_allclose(out.numpy(), out_ref, atol=1e-12)

    def test_sdpa_scaling_matters(self):
        # With d_k = 4 the scale is 1/2: weights must differ from the unscaled softmax.
        q = k = torch.tensor([[[2.0, 0, 0, 0], [0, 0, 0, 0]]])
        _, w = fm.scaled_dot_product_attention(q, k, k)
        unscaled = torch.softmax(q @ k.transpose(-2, -1), -1)
        self.assertFalse(torch.allclose(w, unscaled))
        self.assertAlmostEqual(float(w[0, 0, 0]), float(torch.softmax(torch.tensor([2.0, 0.0]), 0)[0]), places=5)

    def test_model_weights_match_reference_from_model_parameters(self):
        m = make("attention", seed=3)
        x = batch(3, seed=4)
        _, attn = m(x)
        p = {n: t.detach().double().numpy() for n, t in m.named_parameters()}
        h = x.double().numpy() @ p["input_proj.weight"].T + p["input_proj.bias"] + m.pos_enc.double().numpy()
        q = h @ p["w_q.weight"].T + p["w_q.bias"]
        k = h @ p["w_k.weight"].T + p["w_k.bias"]
        s = q @ k.transpose(0, 2, 1) / math.sqrt(m.d_model)
        e = np.exp(s - s.max(-1, keepdims=True))
        np.testing.assert_allclose(attn.detach().numpy(), e / e.sum(-1, keepdims=True), atol=1e-5)

    def test_positional_encoding(self):
        pe = fm.sinusoidal_positional_encoding(72, 8)
        self.assertEqual(pe.shape, (72, 8))
        np.testing.assert_allclose(pe[0].numpy(), [0, 1, 0, 1, 0, 1, 0, 1], atol=1e-7)
        for p in (1, 17, 71):
            for i in range(4):
                f = 10000.0 ** (-2 * i / 8)
                self.assertAlmostEqual(float(pe[p, 2 * i]), math.sin(p * f), places=5)
                self.assertAlmostEqual(float(pe[p, 2 * i + 1]), math.cos(p * f), places=5)
        self.assertEqual(len({tuple(r.tolist()) for r in pe}), 72)  # every hour distinguishable
        with self.assertRaises(ValueError):
            fm.sinusoidal_positional_encoding(10, 7)

    def test_positional_encoding_breaks_permutation_invariance(self):
        m = make("attention")
        x = batch(1)
        out, _ = m(x)
        out_rev, _ = m(x.flip(1))
        self.assertFalse(torch.allclose(out, out_rev, atol=1e-4))

    def test_positional_encoding_is_a_buffer_not_a_parameter(self):
        m = make("attention")
        self.assertNotIn("pos_enc", dict(m.named_parameters()))
        self.assertIn("pos_enc", dict(m.named_buffers()))


class TrainingBehaviourTests(unittest.TestCase):
    def test_gradient_flow_to_every_parameter(self):
        for kind in fm.MODEL_TYPES:
            m = make(kind).train()
            out, _ = m(batch(4))
            torch.nn.functional.l1_loss(out, torch.zeros_like(out)).backward()
            for n, p in m.named_parameters():
                self.assertIsNotNone(p.grad, f"{kind}.{n} has no grad")
                self.assertTrue(torch.isfinite(p.grad).all(), f"{kind}.{n}")
                self.assertGreater(float(p.grad.abs().sum()), 0.0, f"{kind}.{n} grad is all zero")

    def test_gradient_reaches_input(self):
        for kind in fm.MODEL_TYPES:
            x = batch(2).requires_grad_(True)
            out, _ = make(kind)(x)
            out.sum().backward()
            self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_tiny_overfit_reduces_loss(self):
        for kind in fm.MODEL_TYPES:
            torch.manual_seed(0)
            m = fm.build_model(kind)
            x, y = batch(8), torch.randn(8, 24)
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            first = None
            for _ in range(60):
                opt.zero_grad()
                loss = torch.nn.functional.l1_loss(m(x)[0], y)
                first = first if first is not None else float(loss)
                loss.backward()
                opt.step()
            self.assertLess(float(loss), 0.7 * first, kind)

    def test_batch_independence_in_eval_mode(self):
        x = batch(6)
        for kind in fm.MODEL_TYPES:
            m = make(kind)
            with torch.no_grad():
                full, attn = m(x)
                for i in (0, 3, 5):
                    one, attn_one = m(x[i:i + 1])
                    np.testing.assert_allclose(one[0].numpy(), full[i].numpy(), atol=1e-5)
                    if attn is not None:
                        np.testing.assert_allclose(attn_one[0].numpy(), attn[i].numpy(), atol=1e-5)

    def test_state_dict_save_load_equivalence(self):
        x = batch(4)
        for kind in fm.MODEL_TYPES:
            src = make(kind, seed=1)
            buf = io.BytesIO()
            torch.save(src.state_dict(), buf)
            buf.seek(0)
            dst = make(kind, seed=99)  # different init
            with torch.no_grad():
                self.assertFalse(torch.allclose(src(x)[0], dst(x)[0]))
            dst.load_state_dict(torch.load(buf, weights_only=True))
            with torch.no_grad():
                a, wa = src(x)
                b, wb = dst(x)
            np.testing.assert_array_equal(a.numpy(), b.numpy())
            if wa is not None:
                np.testing.assert_array_equal(wa.numpy(), wb.numpy())

    def test_config_rebuilds_same_architecture(self):
        for kind in fm.MODEL_TYPES:
            m = make(kind)
            cfg = dict(m.config())
            self.assertEqual(cfg.pop("model_type"), kind)
            self.assertEqual(fm.count_parameters(fm.build_model(kind, **cfg)), fm.count_parameters(m))

    def test_numpy_float32_windows_feed_models(self):
        # WindowSet.inputs is float32 [N,72,1] numpy; torch.from_numpy must be accepted as-is.
        x = torch.from_numpy(np.zeros((3, 72, 1), dtype=np.float32))
        for kind in fm.MODEL_TYPES:
            self.assertEqual(make(kind)(x)[0].shape, (3, 24))


if __name__ == "__main__":
    unittest.main()
