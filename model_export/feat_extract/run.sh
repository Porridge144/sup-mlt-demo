#!/bin/bash

# . ./cmd.sh || exit 1;
export train_cmd="run.pl"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


do_delta=false
dumpdir=dump

python preprocdir/audio2wavscp.py

feat_tr_dir=preprocdir/${dumpdir}; mkdir -p ${feat_tr_dir}
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 \
                            preprocdir preprocdir/exp/make_fbank preprocdir/fbank

./dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        preprocdir/feats.scp preprocdir/cmvn.ark preprocdir/exp/dump_feats ${feat_tr_dir}