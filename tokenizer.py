import os
import torch
import glob
import argparse
from tokenizers import Tokenizer
import sentencepiece as spm
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing

special_token_dict = {"unknown_token": "[UNK]",
                      "pad_token": "[PAD]", 
                      "start_token": "[BOS]",
                      "end_token": "[EOS]"}

def train_tokenizer(path_to_data_root):

    """
    We need to train a WordPiece tokenizer on our french data (as our regular tokenizers are mostly for English!)
    I set all the special tokens we need above:
        unkown_token: Most important incase tokenizer sees a token not a part of our original token set
        pad_token: Padding for the french text
        start_token: Prepend all french text with start token so the decoder has an input to start generating from
        end_token: Append all french text with end token so decoder knowsn when to stop generating anymore. 

    The only thing in here to keep in mind is the normalizers. There are some issues with how the same letter can 
    be represented in Unicode, so we have to do unicode normalization.

    For example: 

    "é" can be written as either (\u00E9) as a single unicode
    "é" can also be written as "e" + ' where we break the accents off of the e and write as a sequence of 2 unicode characters \u0065\u0301

    We want all our data to be in one or the either for some consistency, so we will be using NMC which tries to represent these characters
    with just a single unicode
    """
    

    spm.SentencePieceTrainer.train(
        input= os.path.join(path_to_data_root, "train.vi.txt"),
        model_prefix='vietnamese_tokenizer',  # <--- THIS IS THE FILENAME
        vocab_size=32000,
        model_type='bpe'
    )

class VienameseTokenizer:
    def __init__(self, tokenizer_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        #specific sentecepeice defaults
        self.bos.id = self.sp.bos_id()
        self.eos.id = self.sp.eos_id()  
        self.unk.id = self.sp.unk_id()
        self.pad.id = self.sp.pad_id()

        if self.pad.id == -1:
            self.pad.id = self.unk.id  # set pad to unk if no pad token

    def encode(self, text, add_bos = True, add_eos = True, return_tensors = None):
        #conver string to list of token ids
        ids = self.sp.encode(text)

        if add_bos and self.bos.id != -1:
            ids = [self.bos.id] + ids
        if add_eos and self.eos.id != -1:
            ids = ids + [self.eos.id]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids
    
    def decode(self, ids, skip_special_tokens = True):
        
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        
        if skip_special_tokens:
            ids = [id for id in ids if id not in {self.bos.id, self.eos.id, self.pad.id, self.unk.id}]
        
        return self.sp.decode(ids)

    def tokenize(self, text):
        return self.sp.encode_as_pieces(text, out_type=str)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
    
    def __len__(self):
        return self.vocab_size


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Prep")

    parser.add_argument(
        "--path_to_data_root", 
        required=True, 
        help="Path to where you want to save the final tokenized dataset",
        type=str
    )

    args = parser.parse_args()

    train_tokenizer(args.path_to_data_root)

    tokenizer = VietnameseTokenizer("trained_tokenizer/french_wp.json")
    sentence = "Héllo world!"
    enc = tokenizer.encode(sentence)
    print(enc)
    dec = tokenizer.decode(enc, skip_special_tokens=False)
    print(dec)
