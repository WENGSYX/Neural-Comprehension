import json
import torch


class Tokenizer:
    def __init__(self, encoding_map,decoding_map):
        self.vocab = encoding_map
        self.decoder_vocab = decoding_map

    def tokenize(self, text,return_tensor='pt'):
        tokens = [self.vocab.get('bos')]
        for word in text.strip().split():
            if word.isdigit():
                word = int(word)
            if word in self.vocab:
                tokens.append(self.vocab[word])
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens

    def decode(self,output):
        texts = [[] * output.size(0)]
        output = output.cpu().tolist()

        for n in range(len(output)):
            texts[n] = [str(self.decoder_vocab[x]) for x in output[n]]
        return texts
