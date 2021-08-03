import torch
import sys
sys.path.append("..")
from src.utils.utility import to_cuda


class BeamHypotheses(object):
    def __init__(self, beam_size, max_len, length_penalty, early_stop=False):
        self.max_len = max_len - 1  # ignoring bos_token
        self.beam_size = beam_size
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty = length_penalty
        self.early_stop = early_stop

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp)**self.length_penalty
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.beam_size:
                sorted_scores = sorted([
                    (s, idx) for idx, (s, _) in enumerate(self.beams)
                ])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        if len(self) < self.beam_size:
            return False
        elif self.early_stop:  #do not update the hypo list even the score is better.
            return True
        else:
            cur_score = best_sum_logprobs / self.max_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def beam_search(model,
                src,
                lang1_id=None,
                lang2_id=None,
                max_len=None,
                beam_size=5,
                length_penalty=0.6,
                early_stop=False):

    bsz = src.size(0)
    generated = torch.LongTensor(max_len.max(), bsz * beam_size).fill_(
        model.pad_index).cuda()
    generated[0] = model.bos_index

    beam_scores = torch.FloatTensor(bsz, beam_size).fill_(0).cuda()
    beam_scores[:, 1:] = -1e9  # assume first word bos always from first beam
    final_scores = beam_scores.clone()
    generated_hyps = [
        BeamHypotheses(beam_size=beam_size,
                       max_len=max_len[i].item(),
                       length_penalty=length_penalty,
                       early_stop=early_stop) for i in range(bsz)
    ]
    max_len = max_len.max()

    model.eval()
    encoder = model.sentenceRep
    decoder = model.target[0]
    cache = {}
    with torch.no_grad():
        cur_len = 1
        encoder_out, _ = encoder(src=src.cuda(),
                                 lang_id=lang1_id)  # bsz x seqlen x dim
        dim = encoder_out.size(-1)
        src_mask = src != model.pad_index
        src_mask = (src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(
            -1, src_mask.size(-1))).unsqueeze(1).cuda()
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            bsz * beam_size, -1, dim)

        done = [False for _ in range(bsz)]
        while cur_len < max_len:
            tensor = decoder(generated[:cur_len].t(),
                             encoder_out,
                             src_mask,
                             lang2_id,
                             cache=cache)  # [s, nxbeam, dim]
            tensor = tensor[:, -1, :]
            scores = torch.log_softmax(decoder.output_layer(tensor).squeeze(0),
                                       dim=-1)  # [n x beam, nwords]
            scores = scores + beam_scores.view(-1, 1)
            scores = scores.view(bsz, -1)  # [n, beam x nwords]
            next_scores, next_words = torch.topk(scores, beam_size * 2, dim=-1)
            n_words = decoder.n_words
            next_batch_beam = []  #(score,  token_id,  batch_beam_idx)

            for sent_idx in range(bsz):

                done[sent_idx] = done[sent_idx] or generated_hyps[
                    sent_idx].is_done(next_scores[sent_idx].max().item())
                if done[sent_idx]:
                    next_batch_beam.extend([(0, model.pad_index, 0)] *
                                           beam_size)
                    continue

                next_sent_beam = []
                for idx, value in zip(next_words[sent_idx],
                                      next_scores[sent_idx]):
                    beam_idx = idx // n_words
                    word_idx = idx % n_words
                    batch_beam_idx = sent_idx * beam_size + beam_idx

                    if word_idx == model.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_idx].add(
                            generated[:cur_len, batch_beam_idx].clone(),
                            value.item())
                    else:
                        next_sent_beam.append(
                            (value, word_idx, batch_beam_idx))

                    if len(next_sent_beam) == beam_size:
                        break

                if len(next_sent_beam) == 0:  #
                    next_sent_beam = [(0, model.pad_index, 0)] * beam_size

                next_batch_beam.extend(next_sent_beam)

            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src.new([x[2] for x in next_batch_beam])
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words

            for k in cache.keys():
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            cur_len += 1

            if all(done):
                break

        tgt_len = src.new(bsz)
        best = []

        for i, hyp in enumerate(generated_hyps):
            best_hyp = max(hyp.beams, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)

        decoded = src.new(tgt_len.max().item(), bsz).fill_(model.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = model.eos_index

    return decoded.t()