import torch
import csv
from pytorch_transformers import BertConfig,AdamW,WarmupLinearSchedule,WarmupConstantSchedule,WarmupCosineSchedule
from args import get_args
from train_evalute import do_train, evaluate
from dataloader import dataloader,get_data
from config import *
# from pretrained.BertOrigin import BertOrigin
# from pretrained.BertCNN import BertCNN

def decode_sentiment(score, include_neutral=True):
    # score between -1~1
    if include_neutral:
        label = 'NEUTRAL'
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = 'NEGATIVE'
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = 'POSITIVE'

        return label
    else:
        return 'NEGATIVE' if score < 0 else 'POSITIVE'

def print_params(model):
    print("Parameters which can be trained")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,param.is_cuda)

def test_predictions(model,test,path,batch_size):
    # The model needs to be in evaluation mode
    model.eval()

    x_test, emoji_test , y_test = test

    test_dataloader = dataloader(test, MAX_SEQ_LENGTH,model.tokenizer,batch_size=batch_size)

    f1, acc, predicted_labels,_ = evaluate(model,dataloader=test_dataloader,device=model.device)
    print('--------------- Test ---------------')
    print(f"----F1 score: {f1}, Accuracy: {acc}--")

    predictions = []
    labels_meaning = {v: k for k, v in TASK_LABELS[model.option].items()}
    predictions += [labels_meaning[p] for p in list(predicted_labels)]

    with open(path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['tweet','prediction','ground_truth'])
        for i, prediction in enumerate(predictions):
            writer.writerow([x_test[i], prediction,y_test[i]])
    return predictions

def main(args):

    class_weights,train, dev, test = get_data(
        option=args.option, dataset_size=args.dataset_size,unbalanced=args.unbalanced)

    option = args.option
    using_GPU = torch.cuda.is_available() and args.using_GPU


    config = BertConfig(num_labels=len(TASK_LABELS[option]),output_attentions = True)
    # config = BertConfig()

    if args.model_type == 'BertOrigin':
        from pretrained.BertOrigin import BertOrigin
        modelcreator = BertOrigin
    elif args.model_type == 'BertCNN':
        from pretrained.BertCNN import BertCNN
        modelcreator = BertCNN
    elif args.model_type == 'BertAttn':
        from pretrained.BertAttn import BertAttn
        modelcreator = BertAttn

    if args.do_train:
    # create and train model
        #BertConfig
        config.from_pretrained(PRETRAINED_WEIGHTS)
        # print('before load',config)
        model = modelcreator(config, option=option, dropout=args.dropout,
            gpu=using_GPU, seed=args.seed, do_lower_case=args.do_lower_case)
        #froze the parameters of bert
        if args.frozen:
            for param in model.bert.parameters():
                param.requires_grad = False
            print_params(model)

        # optimizer and Warmup Schedule
        model_params = list(model.named_parameters())
        # print_params(model)
        #set the weight decay of LayerNorm and bias is zero
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in model_params if not any(nd in name for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [param for name, param in model_params if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
        elif args.optimizer == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate, correct_bias=args.correct_bias)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.5)

        scheduler = None
        if args.warmup_proportion != 0:
            num_total_steps = int(len(train) / args.batch_size) * args.epochs
            # 1.implements AdamW without compensatation for the bias
            # 2.implements weight decay fix
            if args.warmup_schedules == 'linear':
                scheduler = WarmupLinearSchedule(
                    optimizer, warmup_steps=num_total_steps * args.warmup_proportion, t_total=num_total_steps, last_epoch=-1)
            elif args.warmup_schedules == 'constant':
                scheduler = WarmupConstantSchedule(
                    optimizer, warmup_steps=num_total_steps * args.warmup_proportion, t_total=num_total_steps, last_epoch=-1)
            elif args.warmup_schedules == 'cosine':
                scheduler = WarmupCosineSchedule(
                    optimizer, warmup_steps=num_total_steps * args.warmup_proportion, t_total=num_total_steps, cycles=0.5, last_epoch=-1)

        #datasample
        train_dataloader = dataloader(
            train, MAX_SEQ_LENGTH, model.tokenizer, args.batch_size,is_sample=args.sample)
        dev_dataloader = dataloader(
            dev, MAX_SEQ_LENGTH, model.tokenizer, args.batch_size)

        # reload for pretraining
        model.set_focal_loss(alpha=class_weights,gamma=args.gamma)
        model.load_model(args.model_load,args.model_dir)


        model_saved_path = do_train(model, train_dataloader, dev_dataloader, args.epochs,
                                 optimizer, scheduler, args.dataset_size, args.early_stop, args.print_step, args.gradient_accumulation_steps,
                                 args.batch_size, args.learning_rate, model_path=PATH_CONFIG)

        test_predictions(model, test, model_saved_path[:-1] + ".csv",args.batch_size)
    elif args.model_dir:
        config.from_pretrained(PRETRAINED_WEIGHTS)
        model = modelcreator(config, option=option, dropout=args.dropout,
            gpu=using_GPU, seed=args.seed, do_lower_case=args.do_lower_case)

        model.set_focal_loss(alpha=class_weights,gamma=args.gamma)
        model.load_model(args.model_load,args.model_dir)
        # model_dir = "./results/B64_lr1e-05_s0.01_0819_2023/"
        test_predictions(model, test,
                         args.model_dir[:-1] + ".csv", batch_size=args.batch_size)

if __name__ == "__main__":

    PATH_CONFIG = "./results/"
    # PATH_CONFIG = None

    args = get_args()
    print('\n')
    print(args)

    main(args)
