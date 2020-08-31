import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debugmode', type=int, default=0)
parser.add_argument('--model_path', type=str, default="model.last10.avg.best")
parser.add_argument('--recog_json', type=str, default="data.1.json")
parser.add_argument('--preprocess_conf', type=str, default=None)
parser.add_argument('--num_encs', type=int, default=1)
parser.add_argument('--nbest', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=0)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--penalty', type=float, default=0.0)
parser.add_argument('--maxlenratio', type=float, default=0.0)
parser.add_argument('--minlenratio', type=float, default=0.0)
parser.add_argument('--ctc_weight', type=float, default=0.5)
parser.add_argument('--lm_weight', type=float, default=0.3)

parser.add_argument('--result_label', type=str, default="results.json")
args = parser.parse_args()


from espnet.asr.pytorch_backend.asr import recog
recog(args)