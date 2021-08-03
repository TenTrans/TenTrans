import sys
import argparse
import torch
import time
import numpy as np
sys.path.append("..")
from src.search.greedy_search import greedy_search
from src.search.beam_search import beam_search
from src.data.vocab import Vocabulary
from src.utils.utility import batch_data
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.seq2seq_model import Seq2SeqModel
from src.utils.utility import str2bool



def translate(args):
    batch_size = args.batch_size

    src_vocab = Vocabulary(file=args.src_vocab)
    tgt_vocab = Vocabulary(file=args.tgt_vocab)
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_lang_id = src_vocab.index(f"[[{src_lang}]]")
    tgt_lang_id = src_vocab.index(f"[[{tgt_lang}]]")

    checkpoint = torch.load(args.model_path, map_location='cpu')
    config = checkpoint['config']
    output = open(args.output, "w") if args.output else sys.stdout

    sentenceRepModel = SentenceRepModel.build_model(config['sentenceRep'],
                                                    src_vocab)
    model = Seq2SeqModel(config['target'],
                         sentenceRepModel,
                         tgt_vocab=tgt_vocab)
    model.sentenceRep.load_state_dict(checkpoint['model_sentenceRep'])
    model.target.load_state_dict(checkpoint['model_target'])
    model.cuda()
    model.eval()

    print(f"Loading model from epoch_{checkpoint['epoch']}....", file=output)

    src_sent = open(args.src, "r").readlines()
    start = time.perf_counter()

    src_idx = [[
        src_vocab.index(w) for w in s.strip().split()[:args.max_seq_length]
    ] for s in src_sent]
    src_len = np.array([len(s) for s in src_idx])
    idx_by_length = np.argsort(src_len) if args.decode_by_length else np.arange(len(src_len))
    candidate = {}

    ntokens = 0
    for i in range(0, len(src_sent), batch_size):
        src = [src_idx[j] for j in idx_by_length[i:i + batch_size]]

        src = batch_data(src, src_vocab.pad_index, src_vocab.eos_index)

        if args.beam == 1:
            max_len = int((src != src_vocab.pad_index).sum(-1).max() *
                          args.max_len_a + args.max_len_b)
            max_len = min(max_len, args.max_seq_length)
            generated = greedy_search(model,
                                      src=src,
                                      lang1_id=src_lang_id,
                                      lang2_id=tgt_lang_id,
                                      max_len=max_len)
        else:
            max_len = ((src != src_vocab.pad_index).sum(-1) * args.max_len_a +
                       args.max_len_b).int()
            max_len[max_len > args.max_seq_length] = args.max_seq_length
            generated = beam_search(model,
                                    src=src,
                                    lang1_id=src_lang_id,
                                    lang2_id=tgt_lang_id,
                                    beam_size=args.beam,
                                    max_len=max_len,
                                    length_penalty=args.length_penalty,
                                    early_stop=args.early_stop)

        for k, j in enumerate(idx_by_length[i:i + batch_size]):
            hypo = []
            for w in generated[k][1:]:
                if tgt_vocab[w.item()] == '<s>':
                    break
                hypo.append(tgt_vocab[w.item()])
            ntokens += len(hypo)
            hypo = " ".join(hypo)
            src = src_sent[j].strip()
            
            if  args.decode_by_length:
                candidate[j] = (src, hypo)
            else:
                print(f"Source_{j}: {src_sent[j].strip()}", file=output)
                print(f"Target_{j}: {hypo}\n", file=output)


    if args.decode_by_length:
        for k in range(len(candidate)):
            print(f"Source_{k}: {candidate[k][0]}", file=output)
            print(f"Target_{k}: {candidate[k][1]}\n", file=output)

    end = time.perf_counter()
    print(
        f"Total sentences: {len(src_sent)}, decoding cost:{end-start:.2f}, speed: {ntokens/(end-start):.3f} ntokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # for inference
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--src_vocab", type=str, default="")
    parser.add_argument("--tgt_vocab", type=str, default="")
    parser.add_argument("--src_lang", type=str, default="")
    parser.add_argument("--tgt_lang", type=str, default="")
    parser.add_argument("--length_penalty", type=float, default=1)
    parser.add_argument("--max_len_a", type=float, default=1)
    parser.add_argument("--max_len_b", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=500)
    parser.add_argument("--decode_by_length",
                        type=str2bool,
                        nargs='?',
                        default=True,
                        help="if sort the input sentences by lenghths")
    parser.add_argument("--early_stop",
                        type=str2bool,
                        nargs='?',
                        default=False)
    args = parser.parse_args()
    translate(args)
