import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from models.lora import inject_lora, build_model, checkpoint_parts, build_training_model
from types import SimpleNamespace
from models.transformer_cosine_multibatch import TransformerEncoder, TransformerEncoderLayer


def small_model(**kwargs):
    model = nn.Module()
    model.encoder = TransformerEncoder(TransformerEncoderLayer(8, 2, 16, dropout=0),
                                       kwargs.get('num_layers', 2))
    return model


class LoRATest(unittest.TestCase):
    def test_output_gradients_and_batch_independence(self):
        torch.manual_seed(1)
        model = small_model()
        original = copy.deepcopy(model)
        inject_lora(model, rank=2, alpha=2)
        x = torch.randn(4, 2, 8)
        output, features = model.encoder(x, (2, 2))
        expected, expected_features = original.encoder(x, (2, 2))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        for a, b in zip(features, expected_features):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for i in range(2):
            single, _ = model.encoder(x[:, i:i+1], (2, 2))
            torch.testing.assert_close(output[:, i:i+1], single)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            output, features = model.encoder(x, (2, 2))
            (output[..., 0].sum() + sum(f.square().mean() for f in features)).backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                else:
                    self.assertIsNone(p.grad, name)
            optimizer.step()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(p, original.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('.A')))

    @patch('models.vgg_c_multibatch.vgg19_trans', side_effect=small_model)
    def test_resume_file_modes(self, factory):
        args = SimpleNamespace(resume='base.pth', num_layers=2, lora=False,
                               lora_rank=2, lora_alpha=4.0, lora_attention='all')
        base = small_model()
        with patch('torch.load', return_value=base.state_dict()):
            model, config, resume_state = build_training_model(args)
        self.assertIsNone(resume_state)
        self.assertEqual(config['num_layers'], 2)
        self.assertTrue(config['lora'])
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
        checkpoint = dict(model_state_dict=model.state_dict(), model_config=config,
                          optimizer_state_dict=optimizer.state_dict(), epoch=7)
        args.resume = 'run.tar'
        with patch('torch.load', return_value=checkpoint):
            restored, restored_config, resume_state = build_training_model(args)
        self.assertEqual(restored_config, config)
        self.assertEqual(resume_state['epoch'] + 1, 8)
        restored_optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad])
        restored_optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        args.resume = 'adapter.pth'
        with patch('torch.load', return_value=checkpoint), self.assertRaises(ValueError):
            build_training_model(args)
        with patch('torch.load', return_value=small_model(num_layers=4).state_dict()), self.assertRaises(ValueError):
            build_training_model(args)
        args.resume = 'broken.tar'
        with patch('torch.load', return_value=base.state_dict()), self.assertRaises(ValueError):
            build_training_model(args)
        args.resume = 'baseline.tar'
        baseline_checkpoint = dict(model_state_dict=base.state_dict(), epoch=3,
                                   optimizer_state_dict=torch.optim.Adam(base.parameters()).state_dict())
        with patch('torch.load', return_value=baseline_checkpoint):
            baseline, baseline_config, resume_state = build_training_model(args)
        self.assertIsNone(baseline_config['lora'])
        self.assertTrue(all(p.requires_grad for p in baseline.parameters()))

    @patch('models.vgg_c_multibatch.vgg19_trans', side_effect=small_model)
    def test_checkpoint_and_targets(self, factory):
        base = small_model(num_layers=3)
        settings = dict(rank=2, alpha=4.0, attention='qv')
        model, config = build_model(base.state_dict(), lora=settings)
        self.assertEqual(config['num_layers'], 3)
        self.assertEqual(set(model.encoder.layers[0].self_attn.lora_in), {'0', '2'})
        self.assertFalse(hasattr(model.encoder.layers[0].self_attn.out_proj, 'lora'))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('.B'):
                    p.normal_()
        buffer = io.BytesIO()
        torch.save({'model_state_dict': model.state_dict(), 'model_config': config}, buffer)
        buffer.seek(0)
        state, metadata = checkpoint_parts(torch.load(buffer, weights_only=True))
        restored, _ = build_model(state, metadata)
        x = torch.randn(4, 2, 8)
        torch.testing.assert_close(model.encoder(x, (2, 2))[0], restored.encoder(x, (2, 2))[0])
        with self.assertRaises(ValueError):
            build_model(state)
        with self.assertRaises(ValueError):
            build_model(lora=settings)
        broken = dict(base.state_dict())
        del broken['encoder.layers.0.linear1.weight']
        with self.assertRaises(RuntimeError):
            build_model(broken, lora=settings)


if __name__ == '__main__':
    unittest.main()
