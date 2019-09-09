# coding=utf-8

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss, MSELoss
from datetime import datetime,timedelta
from sklearn.metrics import f1_score


def do_train(model, train_dataloader, dev_dataloader,
    epochs, optimizer, scheduler, dataset_size,
    early_stop, print_step, gradient_accumulation_steps,
    batch_size, lr,model_path=None):
    early_stop_times = 0
    best_dev_loss = float('inf')

    device = model.device
    device_count = model.device_count

    print('Training Device: ',device)

    if device_count > 1:
        model = nn.DataParallel(model,device_ids=range(device_count)) # multi-GPU


    #if model_path is None, we are in development mode, all logs and results will not be stored
    if model_path:
        timestamp = (datetime.now()+timedelta(hours=1)).strftime("%m%d_%H%M")
        tb_file='./outputs/'+str(f'B{batch_size}_lr{lr}_s{dataset_size}_{timestamp}')
        print(f'tensorboard saved path: {tb_file}')
        model_saved_path = model_path+str(f'B{batch_size}_lr{lr}_s{dataset_size}_{timestamp}/')
        print(f'model saved path: {model_saved_path}')
        writer = SummaryWriter(tb_file)

    steps = 0
    global_steps = 0

    # The model needs to be in training mode
    model.train()
    optimizer.zero_grad()

    for index_epoch in range(epochs):
        print('-------------------- Epoch: {} --------------------' .format(index_epoch+1))
        if early_stop_times >= early_stop:
            print(early_stop_times)
            # break

        #get batch data
        for step, batch_data in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_data = tuple(t.to(device) for t in batch_data)
            input_ids, input_mask, gnd_labels,emoji_ids, emoji_mask = batch_data

            # Forward pass
            outputs = model(input_ids, attention_mask = input_mask,labels = gnd_labels,emoji_ids = emoji_ids,emoji_mask = emoji_mask)
            train_loss,logits,attention_text = outputs[:3]
            attention_emoji = outputs[-1]

            # Backward pass
            # print(train_loss) shape(2,)
            train_loss = train_loss.mean() # for parallel_model
            train_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if scheduler:
                    scheduler.step()
                #init gradient as zero
                #at each batch, the gradient of loss based on weight is sum of each sample
                optimizer.step()
                optimizer.zero_grad()

                if global_steps % print_step == 0 or step == 0:
                    #save model
                    if model_path:
                        if device_count>1:
                            model.module.save_model(model_saved_path)
                        else:
                            model.save_model(model_saved_path)
                    #calculate development loss
                    f1, acc,_,dev_loss = evaluate(model,dev_dataloader,device)
                    print(f"\n Steps {global_steps} F1 score: {f1:.4f}, Accuracy: {acc:.4f}, train loss : {train_loss.item():.4f}, dev loss : {dev_loss.item():.4f}")

                    if gradient_accumulation_steps > 1:
                        train_loss = train_loss / gradient_accumulation_steps
                        dev_loss = dev_loss / gradient_accumulation_steps

                    if model_path:
                        # for i in range(torch.cuda.device_count()):
                            # print(f'{i}th cuda used {torch.cuda.getMemoryUsage(i)})
                        writer.add_scalars('loss',
                                        {'train': train_loss.item(),
                                        'dev': dev_loss.item()},
                                        global_steps)
                        writer.add_scalars('results',
                                        {'accuracy': acc,
                                        'f1_score': f1},
                                        global_steps)
                    # early stopping
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        early_stop_times = 0
                    else:
                        early_stop_times += 1

                global_steps += 1
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
    if model_path:
        writer.close()
        if device_count>1:
            model.module.save_model(model_saved_path)
        else:
            model.save_model(model_saved_path)
    return model_saved_path

def evaluate(model,dataloader,device):

    f1, acc = 0, 0
    nb_eval_examples = 0

    for batch_data in tqdm(dataloader, desc="DevTest"):
        batch_data = tuple(t.to(device) for t in batch_data)
        input_ids, input_mask, gnd_labels,emoji_ids, emoji_mask = batch_data

        with torch.no_grad():

            outputs = model(input_ids, attention_mask = input_mask,labels = gnd_labels,emoji_ids=emoji_ids, emoji_mask=emoji_mask)
            dev_loss, logits, attention_text = outputs[:3]
            attention_emoji = outputs[-1]

            # Backward pass
            dev_loss = dev_loss.mean()

        predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
        # predicted_labels = torch.argmax(logits,dim=1)
        gnd_labels = gnd_labels.cpu().numpy()
        #calculate accuracy
        tmp_eval_f1 = f1_score(predicted_labels, gnd_labels, average='macro')

        acc+= np.sum(predicted_labels == gnd_labels)
        f1 += tmp_eval_f1 * input_ids.size(0)
        nb_eval_examples += input_ids.size(0)

    return f1 / nb_eval_examples, acc / nb_eval_examples,predicted_labels, dev_loss
