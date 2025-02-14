import torch 
from torch import nn





data = [
    ("hello", "merhaba"),
    ("how are you", "nasılsın"),
    ("good morning", "günaydın"),
    ("thank you", "teşekkür ederim"),
    ("good night", "iyi geceler")
]


def build_vocab(data):
    source_vocab = set()
    target_vocab = set()
    for src, tgt in data:
        source_vocab.update(src.split())
        target_vocab.update(tgt.split())
        
    
    source_vocab = ['<PAD>', '<SOS>', '<EOS>'] + sorted(source_vocab)
    target_vocab = ['<PAD>', '<SOS>', '<EOS>'] + sorted(target_vocab)

    source_word2idx = {idx: word for word, idx in enumerate(source_vocab)}
    target_word2idx = {idx: word for word, idx in enumerate(target_vocab)}      


    source_idx2word = {idx: word for word, idx in source_word2idx.items()}
    target_idx2word = {idx: word for word, idx in target_word2idx.items()}

    return source_word2idx, target_word2idx, source_idx2word, target_idx2word



source_word2idx, target_word2idx, source_idx2word, target_idx2word = build_vocab(data)



def sentence_to_tensor(sentence, word2idx):
    tokens = sentence.split()
    indexed = [word2idx['<SOS>']] + [word2idx[word] for word in tokens] + [word2idx['<EOS>']]
    return torch.tensor(indexed, dtype=torch.long)


train_data = []
for src, tgt in data:
    src_tensor = sentence_to_tensor(src, source_word2idx)
    tgt_tensor = sentence_to_tensor(tgt, target_word2idx)
    train_data.append((src_tensor, tgt_tensor))
    
print("Sample tensor input output:")
print(train_data)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

































