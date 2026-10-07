"""Exercise encoding and effective-input reuse without downloading model weights."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from week6_qwen_hierarchy.qwen_hierarchy import HERE, QwenEncoder


class LastTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        # A deterministic token-specific feature checks pooling and normalization.
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=1024).float() * 3
        return SimpleNamespace(last_hidden_state=hidden)


class EncoderTests(unittest.TestCase):
    def test_pooling_normalization_prompt_and_window_cache(self):
        core = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, 'a': 2, 'b': 3, 'c': 4,
                                           'instruction': 5}, unk_token='[UNK]'))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token='[PAD]', padding_side='left')
        model = LastTokenModel()
        config = json.loads((HERE / 'config/experiment.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(model_cache=Path(temporary), offline=True, dtype='float32', device='cpu',
                embedding_cache=Path(temporary) / 'vectors', batch_size=8, batch_tokens=100)
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('transformers.AutoModel.from_pretrained', return_value=model):
                encoder = QwenEncoder(config, args)
                first = encoder.encode(['a', 'a b c'], 2, 'child')
                self.assertEqual(first.argmax(axis=1).tolist(), [2, 3])
                np.testing.assert_allclose(np.linalg.norm(first, axis=1), [1, 1])
                self.assertEqual(model.calls, 1)
                second = encoder.encode(['a', 'a b c'], 4, 'parent')
                self.assertEqual(second.argmax(axis=1).tolist(), [2, 4])
                self.assertEqual(model.calls, 2)  # Only the formerly truncated input changes.
                encoder.encode(['a', 'a b c'], 8, 'parent')
                self.assertEqual(model.calls, 2)  # Larger window, identical effective inputs.
                self.assertEqual(encoder.audit[0]['truncated_units'], 1)
                with self.assertRaises(ValueError):
                    encoder.encode(['a b'], 2, 'query', prompt='instruction ')
                # New encoder process reuses disk vectors without loading any weights.
                fresh = QwenEncoder(config, args)
                np.testing.assert_array_equal(fresh.encode(['a'], 16, 'child'), first[:1])
                self.assertIsNone(fresh.model)


if __name__ == '__main__':
    unittest.main()
