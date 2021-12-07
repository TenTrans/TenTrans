import argparse
import sys
sys.path.append("..")
from src.search.greedy_search import greedy_search
from src.search.beam_search import beam_search
from src.data.vocab import Vocabulary
from src.utils.utility import batch_data
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.seq2seq_model import Seq2SeqModel
from src.utils.utility import str2bool
import sentencepiece as spm
import torch


class TranslationModel:
    def __init__(self,
                 mt_model,
                 spm_model,
                 mt_src_vocab,
                 mt_tgt_vocab,
                 args,
                 segment_model=None,
                 segment_src_vocab=None):
        self.mt_model = mt_model
        self.spm_model = spm_model
        self.segment_model = segment_model
        self.mt_src_vocab = mt_src_vocab
        self.mt_tgt_vocab = mt_tgt_vocab
        self.segment_src_vocab = segment_src_vocab
        self.args = args

    def tokenize(self, txt):
        txt = self.spm_model.encode(txt, out_type=str)
        return txt

    def translate(self, ocr_text, segment_label="\\n"):
        args = self.args

        ocr_text = [self.tokenize(txt) for txt in ocr_text]
        src_idx = [[
            self.mt_src_vocab.index(w) for w in s[:args.max_seq_length]
        ] for s in ocr_text]
        src = batch_data(src_idx, self.mt_src_vocab.pad_index,
                         self.mt_src_vocab.eos_index)

        if args.beam == 1:
            max_len = int((src != self.mt_src_vocab.pad_index).sum(-1).max() *
                          args.max_len_a + args.max_len_b)
            max_len = min(max_len, args.max_seq_length)
            generated = greedy_search(
                self.mt_model,
                src=src,
                lang1_id=None,
                lang2_id=None,
                max_len=max_len,
            )
        else:
            max_len = (
                (src != self.mt_src_vocab.pad_index).sum(-1) * args.max_len_a +
                args.max_len_b).int()
            max_len[max_len > args.max_seq_length] = args.max_seq_length
            generated = beam_search(
                self.mt_model,
                src=src,
                lang1_id=None,
                lang2_id=None,
                beam_size=args.beam,
                max_len=max_len,
                length_penalty=args.length_penalty,
                early_stop=args.early_stop,
            )
        hypos = []
        for i in range(generated.size(0)):
            hypo = []
            for w in generated[i][1:]:
                if self.mt_tgt_vocab[w.item()] == "<s>":
                    break
                hypo.append(self.mt_tgt_vocab[w.item()])
            hypo = "".join(hypo)
            hypos.append(hypo)

        return hypos


def load_mt_model(model_path, src_vocab_path, tgt_vocab_path):
    src_vocab = Vocabulary(file=src_vocab_path)
    tgt_vocab = Vocabulary(file=tgt_vocab_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    sentence_rep = SentenceRepModel.build_model(config["sentenceRep"],
                                                    src_vocab)
    model = Seq2SeqModel(config["target"],
                         sentence_rep,
                         tgt_vocab=tgt_vocab)
    model.sentence_rep.load_state_dict(checkpoint["model_sentenceRep"])
    model.target.load_state_dict(checkpoint["model_target"])
    model.cuda()
    model.eval()
    return model, src_vocab, tgt_vocab


def translate(args):
    mt_model, src_vocab, tgt_vocab = load_mt_model(args.model, args.src_vocab,
                                                   args.tgt_vocab)
    spm_model = spm.SentencePieceProcessor(model_file=args.spm_model)

    mt_model = TranslationModel(mt_model, spm_model, src_vocab, tgt_vocab,
                                args, None, None)

    while True:
        print("输入> ", end="")
        txt = input()
        results = mt_model.translate([txt])
        print("翻译> ", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--model", type=str, default="")

    parser.add_argument("--src_vocab", type=str, default="")
    parser.add_argument("--tgt_vocab", type=str, default="")

    parser.add_argument("--length_penalty", type=float, default=1)
    parser.add_argument("--max_len_a", type=float, default=1)
    parser.add_argument("--max_len_b", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=500)
    parser.add_argument("--spm_model", type=str, default="")
    parser.add_argument("--early_stop",
                        type=str2bool,
                        nargs="?",
                        default=False)
    args = parser.parse_args()
    translate(args)


main()
