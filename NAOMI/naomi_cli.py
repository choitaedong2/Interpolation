import argparse
from naomi import NAMOIimputation

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='data path')
parser.add_argument('--window', type=int, required=False, default=50, help='imputation window size')
parser.add_argument('--epoch', type=int, required=False, default=200, help='train epoch')
parser.add_argument('--gpu', type=int, required=False, default=0, help='whether to use gpu or not')

args = parser.parse_args()

naomiIMP = NAMOIimputation(args.path, window_size=args.window, use_gpu= args.gpu)
naomiIMP.imputation(200)
naomiIMP.plot()
naomiIMP.df.to_csv("result.csv")
