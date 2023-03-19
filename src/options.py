import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default='cifar10',
                        help="dataset we want to train on")
    
    parser.add_argument('--num_agents', type=int, default=40,
                        help="number of agents:K")
    
    parser.add_argument('--agent_frac', type=float, default=1,
                        help="fraction of agents per round:C")
    
    parser.add_argument('--num_corrupt', type=int, default=4,
                        help="number of corrupt agents")
    
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of communication rounds:R")
    
    parser.add_argument('--aggr', type=str, default='mask_avg',
                        help="aggregation function to aggregate agents' local weights")
    
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs:E")
    
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    
    parser.add_argument('--client_moment', type=float, default=0,
                        help='clients momentum')
    
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")
    
    parser.add_argument('--poison_frac', type=float, default=0.5,
                        help="fraction of dataset to corrupt for backdoor attack")
    
    parser.add_argument('--pattern_type', type=str, default='plus',
                        help="shape of bd pattern")
    
    parser.add_argument('--theta', type=int, default=25,
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--clip', type=float, default=5,
                        help="weight clip to -clip,+clip")
    
    parser.add_argument('--noise', type=float, default=0,
                        help="set noise such that l1 of (update / noise) is this ratio. No noise if 0")
    
    parser.add_argument('--top_frac', type=int, default=100, 
                        help="compare fraction of signs")
    
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")
       
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")
    
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="num of workers for multithreading")
    parser.add_argument('--dense_ratio', type=float, default=0.5,
                        help="num of workers for multithreading")
    parser.add_argument('--anneal_factor', type=float, default=1,
                        help="num of workers for multithreading")
    parser.add_argument('--method', type=str, default="lockdown",
                        help="num of workers for multithreading")
    parser.add_argument('--se_threshold', type=float, default=1e-4,
                        help="num of workers for multithreading")
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--attack',type=str, default="badnet")
    parser.add_argument('--lr_decay',type=float, default= 1)
    parser.add_argument('--mask_init', type=str, default="ERK")
    parser.add_argument('--dis_check_gradient', action='store_true', default=False)
    parser.add_argument('--wd', type=float, default= 1e-4)
    parser.add_argument('--cease_poison', type=float, default=100000)
    args = parser.parse_args()
    return args