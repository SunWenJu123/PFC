import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from utils.args import get_args
from utils.training import train_il
import torch




def main():
    args = get_args()
    args.model = 'pfc'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    args.dataset = 'seq-cifar10'
#    args.print_freq = 10
#    args.n_epochs = 100  
#    args.sche_step = 70
#    
#    args.lr = 0.03
#    args.batch_size = 32 #ok
#    args.ratio = 0.2 #ok
#    args.exp_size = 5000
#    args.gamma = 1


    args.dataset = 'seq-cifar100'
    args.print_freq = 10
    args.n_epochs = 100
    args.sche_step = 80
    
    args.lr = 0.03
    args.batch_size = 32
    args.ratio = 0.2
    args.exp_size = 500
    args.gamma = 1


#    args.dataset = 'seq-tinyimg'
#    args.print_freq = 10
#    args.n_epochs = 100
#    args.sche_step = 45
#    
#    args.lr = 0.02
#    args.batch_size = 16
#    args.ratio = 0.2
#    args.exp_size = 500
#    args.gamma = 1

#    args.dataset = 'seq-tinyimg'
#    args.print_freq = 10
#    args.n_epochs = 100
#    args.sche_step = 45
#    args.nt = 25
#    
#    args.lr = 0.02
#    args.batch_size = 16
#    args.ratio = 0.2
#    args.exp_size = 500
#    args.gamma = 1


    for conf in [1]:
        print("")
        print("=================================================================")
        print("==========================", "", ":", conf, "==========================")
        print("============================s=====================================")
        print("")
        args.repeat = 5
        train_il(args)


if __name__ == '__main__':
    main()
