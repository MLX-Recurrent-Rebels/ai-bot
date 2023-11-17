import sentencepiece as spm


class Tokenizer:
    def __init__(self, prefix='tiny_piece'):
        self.prefix = prefix
        self.sp = spm.SentencePieceProcessor(f"./{self.prefix}.model")

    def encode(self, txt):
        return self.sp.encode_as_ids(txt)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def train(self, path):
        spm.SentencePieceTrainer.Train(
            input=path,
            model_type='bpe',
            model_prefix=self.prefix,
            vocab_size=10000,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[BOS]',
            eos_piece='[EOS]'
        )

        return self

    def load(self):
        self.sp.load(f'./{self.prefix}.model')
        return self

    def vocab_size(self):
        return self.sp.get_piece_size()
