import sys
sys.path.append("..")
from src.utils.utility import to_cuda
import torch


def greedy_search(model, src, lang1_id=None, lang2_id=None, max_len=200):
    """
    greedy seach decoding
    :param model: NMT model
    :param src: token ids.  2-D tensor
    :param lang1_id: 2-D tensor
    :param lang2_id: 2-D tensor
    :param max_len: the max decoding length of the target senteces. int
    :return: predict tokens: 2-D tensor
    """
    bsz = src.size(0)
    generated = torch.LongTensor(max_len,
                                 bsz).fill_(model.pad_index).to(src.device)
    cur_len = 1
    unfinished = generated.new(bsz).long().fill_(1)
    generated[0] = model.bos_index

    model.eval()
    encoder = model.sentenceRep
    decoder = model.target[0]

    generated, src, unfinished = to_cuda(generated, src, unfinished)
    cache = {}
    with torch.no_grad():
        encoder_out, _ = encoder(src=src, lang_id=lang1_id)
        src_mask = (src != model.pad_index).unsqueeze(1)
        while cur_len < max_len:
            tensor = decoder(generated[:cur_len].T,
                             encoder_out,
                             src_mask,
                             lang2_id,
                             cache=cache)
            tensor = tensor[:, -1, :]  # [bsz, dim]
            scores = decoder.output_layer(tensor)  # [bsz, nwords]
            next_words = scores.topk(1, -1)[1].squeeze(1)
            generated[cur_len] = next_words * unfinished + model.pad_index * (
                1 - unfinished)
            unfinished.mul_((next_words.ne(model.eos_index)).long())
            cur_len += 1
            if unfinished.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished.bool(), model.eos_index)
    assert (generated == model.eos_index).sum() == generated.size(1) * 2
    return generated.t()
