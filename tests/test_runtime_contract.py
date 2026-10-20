import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from jev import runtime
from jev.cascade import Policy, decide
from jev.strata_tokenizer import Tokenizer, bytes_to_unicode


class RuntimeContract(unittest.TestCase):
    def test_reference_tokenizer_lossless_and_special_types(self):
        chars=list(bytes_to_unicode().values())
        tok=Tokenizer(chars+['<ctrl>','<user>'],[],[1]*256+[3,4])
        for text in ['', 'Hello!  123', '\u4f60\u597d e\u0301\U0001f600', '\0\n\r\t']:
            self.assertEqual(tok.decode(tok.encode(text)),text)
        self.assertEqual(tok.encode('<ctrl>',parse_special=True),[256])
        self.assertNotEqual(tok.encode('<ctrl>',parse_special=False),[256])
        self.assertEqual(tok.encode('<user>',parse_special=False),[257])

    def test_logits_supplies_batch_axis_and_preserves_fp16_round(self):
        obj=object.__new__(runtime.HeadRuntime)
        obj.mu=torch.zeros((2,5,10240));obj.sd=torch.ones_like(obj.mu)
        raw=np.ones((4,2,5,10240),dtype=np.float16)
        def fake_head(x,counts):
            self.assertEqual(tuple(x.shape),(1,4,2,5,10240))
            self.assertEqual(x.dtype,torch.float32)
            self.assertEqual(counts.tolist(),[2])
            return torch.tensor([[0.,1.]])
        obj.head=fake_head;obj.temperature=None
        probs=obj.probabilities(raw,2)
        self.assertEqual(len(probs),2);self.assertAlmostEqual(sum(probs),1.,places=6)

    def test_incomplete_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runtime.RuntimeMismatch):
                runtime.HeadRuntime({'running_config': {}}, weights_dir=Path(tmp))

    def test_alternate_weights_missing_is_rejected(self):
        obj=object.__new__(runtime.HeadRuntime)
        obj.identity={'head_files':{'best.pt':'0'*64}}
        with tempfile.TemporaryDirectory() as tmp:
            obj.weights_dir=Path(tmp)
            with self.assertRaises(runtime.RuntimeMismatch):obj._check_head_files()

    def test_invalid_readout_falls_back_but_invalid_decode_fails(self):
        item={'options':['yes','no']}
        def bad(_):raise ValueError('incomplete vector')
        row=decide(item,bad,lambda _:{'choice':'yes'},Policy())
        self.assertEqual((row['route'],row['fallback_reason']),('decode','invalid_head'))
        self.assertEqual(decide(item,bad,lambda _:{'choice':'other'},Policy())['route'],'failed')

if __name__=='__main__':unittest.main()
