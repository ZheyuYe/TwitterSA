import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true','True','TRUE','t', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false','False','FALSE','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_args():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--model_type',
                        type=str,
                        required=True,
                        help="which model we are going to use",
                        choices=['BertOrigin','BertCNN','BertAttn'],
                        default=True)
    parser.add_argument('--do_train',
                        type=str2bool,
                        required=True,
                        help="Train or test mode",
                        choices=[True,False],
                        default=True)
    parser.add_argument('--option',
                        required=True,
                        help="which dataset to train, sentiment140 or SemEval2017 for now",
                        choices=['SemEval2017','sentiment140','emoji','feed'])

    # parser.add_argument('--log_dir',type=str,default=None,help="Log file directory")
    parser.add_argument("--do_lower_case",
                    default=False,
                    type=str2bool,
                    choices=[True,False],
                    help="Set this flag if you are using an uncased model.")

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="The batch size",
                        default=32)

    parser.add_argument("-e", "--epochs",
                        type=int,
                        help="Number of epochs",
                        default=5)

    # fixed Dropout probability with 0.1 as BERT used
    parser.add_argument("--dropout",
                        type=float,
                        help="Dropout keep probability",
                        default=0.1)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("-l", "--learning_rate",
                        type=float,
                        help="Learning rate",
                        default=1e-6)

    parser.add_argument('-o','--optimizer',
                        type=str,
                        help="AdamW, fixed with weight decay",
                        choices=['AdamW','Adam','SGD'],
                        default='Adam')

    parser.add_argument('--correct_bias',
                        type=str2bool,
                        help="Can be set to False to avoid correcting bias in Adam",
                        choices=[True,False],
                        default=False)
# 梯度累积
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")


    parser.add_argument('--warmup_proportion',
                        type=float,
                        help="Probability of steps to warmup, default for not use",
                        default=0)

    parser.add_argument('--warmup_schedules',
                        type=str,
                        help="Warmup schedules in gradient clipping",
                        choices=['constant','cosine','linear'],
                        default='linear')

    parser.add_argument('--using_GPU',
                        type=str2bool,
                        help="use GPU or not",
                        choices=[True,False],
                        default=True)

    parser.add_argument('--dataset_size',
                        required=True,
                        type=float,
                        help="proportion of dataset size to train",
                        default=1)

    parser.add_argument("--early_stop",
                        type=int,
                        default=50,
                        help="Early stop when how many steps keep increasing")

    parser.add_argument('--print_step',
                        type=int,
                        default=100,
                        help="How many steps to save model and print loss information")

    parser.add_argument('--model_dir',
                        type=str,
                        help="load model from exited saved path ending with slash",
                        default=None)
    parser.add_argument('--model_load',
                        type=str,
                        help="load model for pretraining or test",
                        default=False)
    parser.add_argument('--gamma',
                        type=float,
                        help="hyperparameter for focal loss, range 0-5",
                        default=0)
    parser.add_argument('--sample',
                        type=str2bool,
                        help="True for weighted sample, False for Focal Loss",
                        choices=[True,False],
                        default=False)
    parser.add_argument('--frozen',
                        type=str2bool,
                        required=True,
                        help="True for froze the bert model",
                        choices=[True,False],
                        default=False)
    parser.add_argument('--unbalanced',
                        type=str2bool,
                        help="True for unbalanced dataset",
                        choices=[True,False],
                        default=False)
    args = parser.parse_args()
    return args
