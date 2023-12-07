import os
import logging
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eval import eval_acc, eval_acc_for_class
import json


class Trainer():
    def __init__(self, params, graph_classifier, data):
        self.graph_classifier = graph_classifier
        self.params = params
        self.data = data

        graph_classifier.reset_parameters()
        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.CrossEntropyLoss()

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0

    def train(self):
        self.reset_training_state()
        true_label = self.data.y
        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            self.graph_classifier.train()
            self.optimizer.zero_grad()
            score = self.graph_classifier(self.data)
            loss = self.criterion(score[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            
            self.graph_classifier.eval()
            with torch.no_grad():
                val_acc = eval_acc(y_true = self.data.y[self.data.val_mask], y_pred = score[self.data.val_mask])
                if val_acc >= self.best_metric:
                    self.best_metric = val_acc
                    self.test_acc = eval_acc(y_true = self.data.y[self.data.test_mask], y_pred = score[self.data.test_mask])
                    self.test_acc_for_class = eval_acc_for_class(y_true = self.data.y[self.data.test_mask], y_pred = score[self.data.test_mask], num_class = self.params.out_dim)
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        if epoch % 100 == 0:
                            logging.info(f"Validation performance didn\'t improve for {self.not_improved_count} epochs")
                
            time_elapsed = time.time() - time_start
            if epoch % 100 == 0:
                logging.info(f'Epoch {epoch} with loss: {loss}, validation accuracy: {val_acc}, best validation accuracy: {self.best_metric} in {time_elapsed}')
        logging.info(f'Test accuracy: {self.test_acc}') 
        logging.info(f'Test accuracy for class: {self.test_acc_for_class}')
        self.test_acc_for_class['all'] = self.test_acc
        with open(os.path.join(self.params.exp_dir, 'test_acc_for_class.json'), 'w') as json_file:
            json.dump(self.test_acc_for_class, json_file)

    #def save_classifier(self):
    #    torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        
