# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math

import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer


import numpy
import six
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.e2e_asr_common import end_detect

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def recognize(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                               ignore_label=self.ignore_id)

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = x.unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def forward1(self, x):
        # enc_output = self.encode(x).unsqueeze(0)
        lpz = self.ctc.log_softmax(x)
        return lpz

    def forward(self, x):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        ctc_weight  = 0.5
        beam_size   = 1
        penalty     = 0.0
        maxlenratio = 0.0
        nbest       = 1
        sos = eos   = 7442

        import onnxruntime

        # enc_output = self.encode(x).unsqueeze(0)
        encoder_rt = onnxruntime.InferenceSession("encoder.onnx")
        ort_inputs = {"x": x.numpy()}
        enc_output = encoder_rt.run(None, ort_inputs)
        enc_output = torch.tensor(enc_output)
        enc_output = enc_output.squeeze(0)
        # print(f"enc_output shape: {enc_output.shape}")

        # lpz = self.ctc.log_softmax(enc_output)
        # lpz = lpz.squeeze(0)
        ctc_softmax_rt = onnxruntime.InferenceSession("ctc_softmax.onnx")
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

        # initialize hypothesis
        hyp = {'score': 0.0, 'yseq': [y]}


        ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, eos, numpy)
        hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
        hyp['ctc_score_prev'] = 0.0

        # pre-pruning based on attention scores
        # ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
        ctc_beam = 1


        hyps = [hyp]
        ended_hyps = []

        decoder_fos_rt = onnxruntime.InferenceSession("decoder_fos.onnx")

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)

                ort_inputs = {"ys": ys.numpy(), "ys_mask": ys_mask.numpy(), "enc_output": enc_output.numpy()}
                local_att_scores = decoder_fos_rt.run(None, ort_inputs)
                local_att_scores = torch.tensor(local_att_scores[0])
                # local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]


                local_scores = local_att_scores
                # print(local_scores.shape) 1, 7443
                local_best_scores, local_best_ids = torch.topk(
                    local_att_scores, ctc_beam, dim=1)
                # print(local_best_scores.shape) 1, 1
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


    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
