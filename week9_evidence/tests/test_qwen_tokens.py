import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from week9_evidence.qwen import tokenize_inputs


class QwenTokenTests(unittest.TestCase):
    def test_truncation_preserves_template_suffix_not_chat_eos(self):
        raw = Tokenizer(models.WordLevel({'[UNK]': 0, 'a': 1, 'b': 2, 'c': 3,
            '<|endoftext|>': 4, '<|im_end|>': 5}, unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        raw.post_processor = processors.TemplateProcessing(single='$A <|endoftext|>',
            special_tokens=[('<|endoftext|>', 4)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
            eos_token='<|im_end|>', additional_special_tokens=['<|endoftext|>'])
        full, clipped = tokenize_inputs(tokenizer, ['a b c', 'a'], 3)
        self.assertEqual(full[0], [1, 2, 3, 4])
        self.assertEqual(clipped[0], [1, 2, 4])
        self.assertNotEqual(full[0][:3][-1], clipped[0][-1])
        self.assertEqual(clipped[1], [1, 4])
        self.assertNotEqual(clipped[0][-1], tokenizer.eos_token_id)


if __name__ == '__main__':
    unittest.main()
