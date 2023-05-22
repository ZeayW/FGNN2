import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input',type=int,default=2)
    parser.add_argument('--save_dir',type=str,default='../truthtables')
    parser.add_argument('--data_dir', type=str, default='../truthtables/i')
    parser.add_argument('--num_replace',type=int,default=3)
    options = parser.parse_args(args)

    return options
