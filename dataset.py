import torch
from datasets import load_dataset
import tokenizer  # Assuming you have a tokenizer module

class TinyOrcaDataset(torch.utils.data.Dataset):
    def __init__(self):
        ds = load_dataset("nampdn-ai/tiny-orca-textbooks", 'r')
        self.ds = ds
        self.tknz = tokenizer.Tokenizer()

    def __len__(self):
        return len(self.ds['train'])  # Use 'train' directly

    def __getitem__(self, idx):
        entry = self.ds['train'][idx]  # Access 'train' directly
        textbook, question, response = entry['textbook'], entry['question'], entry['response']

        input_text = [self.tknz.stoi['<sos>']] + self.tknz.encode(textbook)
        label_text = input_text[1:] + [self.tknz.stoi['<eos>']]
        masks = [1] * len(input_text)

        return {
            'textbook': textbook,
            'question': question,
            'response': response,
            'input': torch.tensor(input_text),
            'label': torch.tensor(label_text),
            'masks': torch.tensor(masks),
        }

    def collate_fn(self, batch):
        input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True, padding_value=0)
        label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)
        masks_pad = torch.nn.utils.rnn.pad_sequence([item['masks'] for item in batch], batch_first=True, padding_value=0)

        return {
            'textbook': [item['textbook'] for item in batch],
            'question': [item['question'] for item in batch],
            'response': [item['response'] for item in batch],
            'input': input_pad,
            'label': label_pad,
            'masks': masks_pad,
        }

if __name__ == '__main__':
    ds = TinyOrcaDataset()
    print('len(ds)', len(ds))
    print('ds[0]', ds[0])

