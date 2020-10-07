import torch

import onnxruntime

import numpy
import six

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

import json
import kaldiio

import onnxruntime

import os

# load info into memory
main_dir = os.getcwd() + "/"
server_dir = os.path.join(os.getcwd(), "../server") + "/"

with open(main_dir + "charlist.json", 'rb') as f:
    char_list = json.load(f)['char_list']
    f.close()

encoder_rt = onnxruntime.InferenceSession(main_dir + "encoder.onnx")
ctc_softmax_rt = onnxruntime.InferenceSession(main_dir + "ctc_softmax.onnx")
decoder_fos_rt = onnxruntime.InferenceSession(main_dir + "decoder_fos.onnx")


def infer(x, encoder_rt, ctc_softmax_rt, decoder_fos_rt):

    ctc_weight  = 0.5
    beam_size   = 1
    penalty     = 0.0
    maxlenratio = 0.0
    nbest       = 1
    sos = eos   = 7442


    # enc_output = self.encode(x).unsqueeze(0)
    ort_inputs = {"x": x.numpy()}
    enc_output = encoder_rt.run(None, ort_inputs)
    enc_output = torch.tensor(enc_output)
    enc_output = enc_output.squeeze(0)
    # print(f"enc_output shape: {enc_output.shape}")

    # lpz = self.ctc.log_softmax(enc_output)
    # lpz = lpz.squeeze(0)
    ort_inputs = {"enc_output": enc_output.numpy()}
    lpz = ctc_softmax_rt.run(None, ort_inputs)
    lpz = torch.tensor(lpz)
    lpz = lpz.squeeze(0).squeeze(0)
    # print(f"lpz shape: {lpz.shape}")

    h = enc_output.squeeze(0)
    # print(f"h shape: {h.shape}")
    
    # preprare sos
    y = sos
    maxlen = h.shape[0]
    minlen = 0.0
    ctc_beam = 1

    # initialize hypothesis
    hyp = {'score': 0.0, 'yseq': [y]}
    ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, eos, numpy)
    hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
    hyp['ctc_score_prev'] = 0.0

    # pre-pruning based on attention scores
    hyps = [hyp]
    ended_hyps = []


    for i in six.moves.range(maxlen):
        hyps_best_kept = []
        for hyp in hyps:

            # get nbest local scores and their ids
            ys_mask = subsequent_mask(i + 1).unsqueeze(0)
            ys = torch.tensor(hyp['yseq']).unsqueeze(0)

            ort_inputs = {"ys": ys.numpy(), "ys_mask": ys_mask.numpy(), "enc_output": enc_output.numpy()}
            local_att_scores = decoder_fos_rt.run(None, ort_inputs)
            local_att_scores = torch.tensor(local_att_scores[0])

            local_scores = local_att_scores
            local_best_scores, local_best_ids = torch.topk(
                local_att_scores, ctc_beam, dim=1)
            ctc_scores, ctc_states = ctc_prefix_score(
                hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
            local_scores = \
                (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
            local_best_scores, joint_best_ids = torch.topk(local_scores, beam_size, dim=1)
            local_best_ids = local_best_ids[:, joint_best_ids[0]]


            for j in six.moves.range(beam_size):
                new_hyp = {}
                new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])


                new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                # will be (2 x beam) hyps at most
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(
                hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam_size]

        # sort and get nbest
        hyps = hyps_best_kept

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            for hyp in hyps:
                hyp['yseq'].append(eos)

        # add ended hypothes to a final list, and removed them from current hypothes
        # (this will be a probmlem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == eos:
                # only store the sequence that has more than minlen outputs
                # also add penalty
                if len(hyp['yseq']) > minlen:
                    hyp['score'] += (i + 1) * penalty
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)

        # end detection
        if end_detect(ended_hyps, i) and maxlenratio == 0.0:
            break

        hyps = remained_hyps
        if len(hyps) > 0:
            pass
        else:
            break

    nbest_hyps = sorted(
        ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]

    # return nbest_hyps
    return torch.tensor(nbest_hyps[0]['yseq'])


def scp_modify_handler():
    f = open(main_dir + "feat_extract/preprocdir/dump/feats.scp", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        file_id = line.strip().split()[0]
        path_ark = line.strip().split()[1]
        feat = kaldiio.load_mat(path_ark)
        feat = torch.from_numpy(feat)
        # print(f"input size: {feat.shape}")
        nbest_hyps = infer(feat, encoder_rt, ctc_softmax_rt, decoder_fos_rt)
        # get token ids and tokens
        tokenid_as_list = list(map(int, nbest_hyps[1:]))
        token_as_list = [char_list[idx] for idx in tokenid_as_list]
        print(token_as_list)
        token_as_list = token_as_list[:-1]
        token_as_list.insert(1, ",")
        f = open(server_dir + file_id + "_decode_output.txt", "w")
        f.write("".join(token_as_list) + "\n")
        f.close()


import pyinotify

class MyEventHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_WRITE(self, event):
        # print("CLOSE_WRITE event:", event.pathname)
        scp_modify_handler()

# watch manager
wm = pyinotify.WatchManager()
wm.add_watch(main_dir + "feat_extract/preprocdir/dump/feats.scp", pyinotify.ALL_EVENTS, rec=True)

# event handler
eh = MyEventHandler()

# notifier
notifier = pyinotify.Notifier(wm, eh)
notifier.loop()



