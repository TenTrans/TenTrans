import torch
import collections
import sys
import argparse
import os


def average_checkpint(inputs):

    num_models = len(inputs)
    print(inputs)
    rep_params = {}
    target_params = {}
    rep_params_keys = None
    target_params_keys = None
    new_state = None
    for path in inputs:

        state = torch.load(path, map_location="cpu")
        if new_state is None:
            new_state = state

        if rep_params_keys is None:
            rep_params_keys = list(state['model_sentenceRep'].keys())

        if target_params_keys is None:
            target_params_keys = list(state['model_target'].keys())

        for k in rep_params_keys:
            if k not in rep_params:
                rep_params[k] = state['model_sentenceRep'][k].clone()
            else:
                rep_params[k] += state['model_sentenceRep'][k]

        for k in target_params_keys:
            if k not in target_params:
                target_params[k] = state['model_target'][k].clone()
            else:
                target_params[k] += state['model_target'][k]

    averaged_rep_params = collections.OrderedDict()
    for k, v in rep_params.items():
        averaged_rep_params[k] = v
        if averaged_rep_params[k].is_floating_point():
            averaged_rep_params[k].div_(num_models)
        else:
            averaged_rep_params[k] //= num_models
    new_state["model_sentenceRep"] = averaged_rep_params

    averaged_target_params = collections.OrderedDict()
    for k, v in target_params.items():
        averaged_target_params[k] = v
        if averaged_target_params[k].is_floating_point():
            averaged_target_params[k].div_(num_models)
        else:
            averaged_target_params[k] //= num_models
    new_state["model_target"] = averaged_target_params

    return new_state


def main():
    parser = argparse.ArgumentParser("")
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+')
    parser.add_argument('--output', required=True, metavar='FILE')

    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"overwrite {args.output}")

    state = average_checkpint(args.inputs)
    torch.save(state, args.output)


main()