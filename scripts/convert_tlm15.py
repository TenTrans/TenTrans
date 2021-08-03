import sys

sys.path.append("..")
from src.data.vocab import Vocabulary
from src.models.encoder.transformer_encoder import TransformerEncoder
import torch
from src.models.encoder.sentence_rep import SentenceRepModel

model_cfg = {
    'type': 'xlm',
    'hidden_size': 1024,
    'ff_size': 4096,
    'num_heads': 8,
    'encoder_layers': 12,
    'dropout': 0.1,
    'embedd_size': 1024,
    'attention_dropout': 0.1,
    'num_lang': 2,
    'activation': 'gelu',
    'learned_pos': True,
    'use_langembed': True
}

vocab = Vocabulary(
    file=
    "/apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/eyi_32w/vocab_xnli_15.v2",
    max_vocab=95000)
model = TransformerEncoder(model_cfg, vocab)

xlm_model = torch.load(
    "/apdcephfs/share_1157259/users/baijunji/data/model/xlm/mlm_tlm_xnli15_1024.pth",
    map_location='cpu')

#load word embedding
hits = 0
for i in range(len(vocab)):
    if vocab[i] in xlm_model['dico_word2id']:
        model.embedding.weight[i] = xlm_model['model']['embeddings.weight'][
            xlm_model['dico_word2id'][vocab[i]]]
        hits += 1
print(hits)
for lang in ['ar', 'bg', 'de', 'el', 'en', 'es', 'zh']:
    lang_embedd = f"[[{lang}]]"
    if vocab.index(lang_embedd, no_unk=False) != vocab.unk_index:
        model.embedding.weight[vocab.index(lang_embedd)] = xlm_model['model'][
            'lang_embeddings.weight'][xlm_model['params']['lang2id'][lang]]
        print(f"find {lang} embedding")

model.pe.pe.weight.requires_grad = False
model.pe.pe.weight.copy_(xlm_model['model']['position_embeddings.weight'])
model.pe.pe.weight.requires_grad = True

model.embed_norm.weight.requires_grad = False
model.embed_norm.bias.requires_grad = False
model.embed_norm.weight.copy_(xlm_model['model']['layer_norm_emb.weight'])
model.embed_norm.bias.copy_(xlm_model['model']['layer_norm_emb.bias'])
model.embed_norm.weight.requires_grad = True
model.embed_norm.bias.requires_grad = True

for layer in range(model_cfg['encoder_layers']):

    model.layers[layer].src_src_att.k_layer.weight.requires_grad = False
    model.layers[layer].src_src_att.v_layer.weight.requires_grad = False
    model.layers[layer].src_src_att.q_layer.weight.requires_grad = False
    model.layers[layer].att_layer_norm.weight.requires_grad = False
    model.layers[layer].ffn_layer_norm.weight.requires_grad = False
    model.layers[layer].src_src_att.output_layer.weight.requires_grad = False
    model.layers[layer].feed_forward.ffn_layer[0].weight.requires_grad = False
    model.layers[layer].feed_forward.ffn_layer[2].weight.requires_grad = False

    model.layers[layer].src_src_att.k_layer.bias.requires_grad = False
    model.layers[layer].src_src_att.v_layer.bias.requires_grad = False
    model.layers[layer].src_src_att.q_layer.bias.requires_grad = False
    model.layers[layer].att_layer_norm.bias.requires_grad = False
    model.layers[layer].ffn_layer_norm.bias.requires_grad = False
    model.layers[layer].src_src_att.output_layer.bias.requires_grad = False
    model.layers[layer].feed_forward.ffn_layer[0].bias.requires_grad = False
    model.layers[layer].feed_forward.ffn_layer[2].bias.requires_grad = False

    model.layers[layer].src_src_att.k_layer.weight.copy_(
        xlm_model['model']['attentions.{}.k_lin.weight'.format(layer)])
    model.layers[layer].src_src_att.v_layer.weight.copy_(
        xlm_model['model']['attentions.{}.v_lin.weight'.format(layer)])
    model.layers[layer].src_src_att.q_layer.weight.copy_(
        xlm_model['model']['attentions.{}.q_lin.weight'.format(layer)])
    model.layers[layer].src_src_att.output_layer.weight.copy_(
        xlm_model['model']['attentions.{}.out_lin.weight'.format(layer)])
    model.layers[layer].att_layer_norm.weight.copy_(
        xlm_model['model']['layer_norm1.{}.weight'.format(layer)])
    model.layers[layer].ffn_layer_norm.weight.copy_(
        xlm_model['model']['layer_norm2.{}.weight'.format(layer)])
    model.layers[layer].feed_forward.ffn_layer[0].weight.copy_(
        xlm_model['model']['ffns.{}.lin1.weight'.format(layer)])
    model.layers[layer].feed_forward.ffn_layer[2].weight.copy_(
        xlm_model['model']['ffns.{}.lin2.weight'.format(layer)])

    model.layers[layer].src_src_att.k_layer.bias.copy_(
        xlm_model['model']['attentions.{}.k_lin.bias'.format(layer)])
    model.layers[layer].src_src_att.v_layer.bias.copy_(
        xlm_model['model']['attentions.{}.v_lin.bias'.format(layer)])
    model.layers[layer].src_src_att.q_layer.bias.copy_(
        xlm_model['model']['attentions.{}.q_lin.bias'.format(layer)])
    model.layers[layer].src_src_att.output_layer.bias.copy_(
        xlm_model['model']['attentions.{}.out_lin.bias'.format(layer)])
    model.layers[layer].att_layer_norm.bias.copy_(
        xlm_model['model']['layer_norm1.{}.bias'.format(layer)])
    model.layers[layer].ffn_layer_norm.bias.copy_(
        xlm_model['model']['layer_norm2.{}.bias'.format(layer)])
    model.layers[layer].feed_forward.ffn_layer[0].bias.copy_(
        xlm_model['model']['ffns.{}.lin1.bias'.format(layer)])
    model.layers[layer].feed_forward.ffn_layer[2].bias.copy_(
        xlm_model['model']['ffns.{}.lin2.bias'.format(layer)])

    model.layers[layer].src_src_att.k_layer.bias.requires_grad = True
    model.layers[layer].src_src_att.v_layer.bias.requires_grad = True
    model.layers[layer].src_src_att.q_layer.bias.requires_grad = True
    model.layers[layer].att_layer_norm.bias.requires_grad = True
    model.layers[layer].ffn_layer_norm.bias.requires_grad = True
    model.layers[layer].src_src_att.output_layer.bias.requires_grad = True
    model.layers[layer].feed_forward.ffn_layer[0].bias.requires_grad = True
    model.layers[layer].feed_forward.ffn_layer[2].bias.requires_grad = True

    model.layers[layer].src_src_att.k_layer.weight.requires_grad = True
    model.layers[layer].src_src_att.v_layer.weight.requires_grad = True
    model.layers[layer].src_src_att.q_layer.weight.requires_grad = True
    model.layers[layer].att_layer_norm.weight.requires_grad = True
    model.layers[layer].ffn_layer_norm.weight.requires_grad = True
    model.layers[layer].src_src_att.output_layer.weight.requires_grad = True
    model.layers[layer].feed_forward.ffn_layer[0].weight.requires_grad = True
    model.layers[layer].feed_forward.ffn_layer[2].weight.requires_grad = True

rep = SentenceRepModel(model)

torch.save(
    {
        'model_sentenceRep': rep.state_dict(),
        'config': {
            'sentenceRep': model_cfg
        }
    },
    "/apdcephfs/share_1157259/users/baijunji/data/model/xlm_tentrans/model_xlm_tlm15.tt"
)
