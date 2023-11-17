import sentencepiece as spm

class Tokenizer:
    def __init__(self):
        self.vocab = ['<pad>', '<eos>', '<sos>']
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, name):
        return [self.stoi[c] for c in name]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens if self.itos[t] not in ('<sos>', '<eos>', '<pad>')])

    def train(self, path):
        spm.SentencePieceTrainer.Train(
            input=path,
            model_type='bpe',
            model_prefix='tiny_piece',
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

        self.save_model()  # Save the trained model

        return self

    def save_model(self):
        # Save the SentencePiece model
        self.sp.save(f'./tiny_piece.model')

    def load(self):
        self.sp = spm.SentencePieceProcessor(f"./tiny_piece.model")
        return self

    def vocab_size(self):
        return self.sp.get_piece_size()


if __name__ == '__main__':
    tknz = Tokenizer()

    # Assuming your dataset is in a different module called 'dataset'
    from dataset import TinyOrcaDataset

    # Specify the path to your dataset
    dataset_path = 'nampdn-ai/tiny-orca-textbooks'

    # Train the tokenizer on your dataset
    tknz.train(dataset_path)

    # Load the trained model
    tknz.load()

    # Rest of your code remains unchanged
    print("tknz.vocab_size()", tknz.vocab_size())
    print('tknz.sp.bos_id()', tknz.sp.bos_id())
    print('tknz.sp.pad_id()', tknz.sp.pad_id())
    print('tknz.sp.eos_id()', tknz.sp.eos_id())
    print('tknz.sp.unk_id()', tknz.sp.unk_id())

    ids_foo = tknz.encode('hello my name is Bes')
    ids_bar = tknz.encode('ciao il mio nome è Bes')
    ids_zoo = tknz.encode('emma')
    print('ids_foo', ids_foo)
    print('ids_bar', ids_bar)
    print('ids_zoo', ids_zoo)
    txt_foo = tknz.decode(ids_foo)
    txt_bar = tknz.decode(ids_bar)
    txt_zoo = tknz.decode(ids_zoo)
    print('txt_foo', txt_foo)
    print('txt_bar', txt_bar)
    print('txt_zoo', txt_zoo)
    for id in range(4): print(id, tknz.sp.id_to_piece(id), tknz.sp.is_control(id))
