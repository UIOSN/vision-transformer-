
from my_train import main
import argparse

if __name__=='__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lrf',type=float,default=0.01)
    parser.add_argument('--weights',type=str,default='',
                        help='初始化权重路径！')
    parser.add_argument('--device',default='cuda:0',help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    
    main(opt)
