import argparse
import sys
sys.path.append("..")
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.classification_model import ClassificationModel
from src.utils.utility import batch_data, truncate, to_cuda
from src.data.vocab import Vocabulary
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--vocab", type=str, default="")
parser.add_argument("--src", type=str)
parser.add_argument("--lang", type=str)
parser.add_argument("--threhold", type=float, default=-1)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--max_vocab", type=int, default=1000000)
args = parser.parse_args()

reloaded = torch.load(args.model)
config = reloaded["config"]

vocab = Vocabulary(file=args.vocab, max_vocab=args.max_vocab)

sentence_rep = SentenceRepModel.build_model(config["sentenceRep"], vocab)
model = ClassificationModel(config["target"], sentenceRep, vocab)

model.sentence_rep.load_state_dict(reloaded["model_sentenceRep"])
model.target.load_state_dict(reloaded["model_target"])
model.cuda()
model.eval()
langid = vocab.index(f"[[{args.lang}]]")
src_sent = open(args.src, "r").readlines()
batch_size = args.batch_size

predicts = []

with torch.no_grad():
    for i in range(0, len(src_sent), batch_size):
        src_ids = [vocab.encode(sent) for sent in src_sent[i : i + batch_size]]
        src_ids = batch_data(src_ids, vocab.pad_index, vocab.eos_index)
        src_ids = truncate(
            src_ids,
            max_seq_length=args.max_seq_length,
            pad_index=vocab.pad_index,
            eos_index=vocab.eos_index,
        )
        langids = torch.tensor([langid] * len(src_ids), dtype=torch.long)
        src_ids, langids = to_cuda(src_ids, langids)
        logits = model(src_ids, langids)
        probs = logits.softmax(dim=-1)
        # if args.threhold > 0:
        #    mask = probs[:, 1] < args.threhold
        #    probs[:, 1][mask] = 0
        scores, predict = probs.topk(k=1, dim=-1)
        length = len(scores)
        for i in range(length):
            print(f"{predict[i].item()} {probs[i][1].item()}")
        # for label in predict.view(-1).cpu().tolist():
        # print(label)
