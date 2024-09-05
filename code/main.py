import torch
import warnings
import argparse
from pathlib import Path

from data import Data
from utils import check_path
from models import GsRCL, RMCL, AFRCL


def main(args):
    torch.manual_seed(args.seed)
    check_path(args.res_path)

    if args.method != 'RMCL':
        args.top_n_genes = None
        warnings.warn('top_n_genes is set to None for all methods except RMCL')
    
    file_X = Path(args.X)
    file_y = Path(args.y)
    data = Data(file_X, file_y, args.name, args.train_size, args.seed, args.cv, args.top_n_genes)
    data.split()

    device = torch.device(f'cuda:{args.gpu}')

    if args.method == 'GsRCL':
        enc_kwargs = {
            'enc_in_dim': data.dim, 
            'enc_dim': 1024, 
            'enc_out_dim': 512, 
            'proj_dim': 256, 
            'proj_out_dim': 128
        }
        model = GsRCL(
            data, args.val_metric, args.loss, args.epochs, args.batch_size, args.step, args.temperature, args.lr, 
            args.wd, device, args.res_path, enc_kwargs
        )
        model.fit_cv()
        torch.save(model.cv_res, 
                   Path(args.res_path, f'{args.method}--{args.loss}--{data.name}--{args.val_metric}--cv_res.pt')
        )
    
    elif args.method == 'RMCL':
        enc_kwargs = {
            'enc_in_dim': data.dim, 
            'enc_dim': 1024, 
            'enc_out_dim': 512, 
            'proj_dim': 256, 
            'proj_out_dim': 128,
            'dropout': args.rm_dropout
        }
        model = RMCL(
            data, args.val_metric, args.loss, args.epochs, args.batch_size, args.step, args.temperature, args.lr, 
            args.wd, device, args.res_path, enc_kwargs
        )
        model.fit_cv()
        torch.save(model.cv_res, 
                   Path(args.res_path, f'{args.method}--{args.loss}--{data.name}--{args.val_metric}--cv_res.pt')
        )

    elif args.method == 'AFRCL':
        enc_kwargs = {
            'enc_in_dim': data.dim, 
            'enc_dim': 1024, 
            'enc_out_dim': 512, 
            'proj_dim': 256, 
            'proj_out_dim': 128
        }
        model = AFRCL(
            data, args.val_metric, args.loss, args.epochs, args.batch_size, args.step, args.temperature, args.lr, 
            args.wd, device, args.res_path, enc_kwargs
        )
        model.fit_cv()
        torch.save(model.cv_res, 
                   Path(args.res_path, f'{args.method}--{args.loss}--{data.name}--{args.val_metric}--cv_res.pt')
        )

    else:
        raise NotImplementedError(
            f'{args.method} is not implemeneted. Please enter one of the following methods: '\
            'GsRCL, RMCL, AFRCL'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--X', help='the full path to file X (i.e. genes expression matrix')
    parser.add_argument('--y', help='the full path to file y (i.e. cell types annotations)')
    parser.add_argument('--name', help='the dataset name')
    parser.add_argument('--method', help='the method name. Please enter one of the following methods: GsRCL, RMCL, or AFRCL')
    parser.add_argument('--res_path', help='the path where the results are saved')
    parser.add_argument('--loss', help='the contrastive loss, either SupCon, SimCLR, or ModifiedSupCon')
    parser.add_argument('--val_metric', help='the metric for model selection, it should be either acc, mcc, or f1')
    parser.add_argument('--train_size', type=float, default=0.8, help='the training set size when splitting the data')
    parser.add_argument('--cv', type=int, default=5, help='the number of cross validation folds')
    parser.add_argument('--top_n_genes', type=int, default=5000, help='the top n highly variable genes (HVGs)')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size. To use all training samples pass the value -1')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for contrastive learning')
    parser.add_argument('--step', type=int, default=5, help='number of steps to measure validation performance for contrastive learning')
    parser.add_argument('--rm_dropout', type=float, default=0.9, help='dropout for the random genes masking method')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature hyperparameter')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu device number')

    args = parser.parse_args()
    main(args)
