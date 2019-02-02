import time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from callbacks import ReduceLROnPlateau
from sklearn.metrics import log_loss, accuracy_score

N_CLASSES = 10

        
class CameraArchitecture(nn.Module):
    def __init__(self, arch, weights_fp=None, pretrained=False):
        super().__init__()
        
        if weights_fp:
            model = arch()
            model.load_weights(weights_fp)
        elif pretrained:
            model = arch(pretrained)
        else:
            model = arch()
        
        n_features = model.fc.in_features
        model.fc = nn.Dropout(0.0)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES)
        )
        self._model = model
    

    def forward(self, x):
        x = self._model(x)
        x = self.fc(x)
        return x
    
    
    def save_model(self, fn):
        torch.save(self.state_dict(), fn)
    
    
    def load_model(self, fn):
        state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)


class CameraModel:
    def __init__(self, model, learning_rate=1e-4, run_parallel=False):
        self.run_parallel = run_parallel
        
        self.model = nn.DataParallel(model) if self.run_parallel else model
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                           factor=0.5, patience=5, min_lr=1e-8, 
                                           min_delta=1e-5, verbose=1)
        self._criterion = nn.CrossEntropyLoss()
    
    
    def scheduler_step(self, loss, epoch):
        self.scheduler.step(loss, epoch)
    
    
    def set_train_mode(self):
        self.model.train()
    
    
    def set_predict_mode(self):
        self.model.eval()
    
    
    @staticmethod
    def ewa(x, x_prev, beta):
        if x_prev:
            return beta * x + (1 - beta) * x_prev
        else:
            return x
        
    
    def train_on_batch(self, X, y):
        if self.run_parallel:
            X = X.cuda()
            y = y.cuda()
        
        y_pred = self.model(X)
        
        loss = self._criterion(y_pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred.cpu().data.numpy()
        
        
    def predict_on_batch(self, X):
        if self.run_parallel:
            X = X.cuda()
        with torch.no_grad():
            y_pred = self.model(X)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred.cpu().data.numpy()
    
    
    def fit(self, data_loader):
        self.set_train_mode()
        batches_per_epoch = len(data_loader)
        n_batches = data_loader.dataset.__len__() // data_loader.batch_size
        mean_acc = None
        mean_loss = None
        start_time = time.time()
        for batch_num, (X, y) in enumerate(data_loader):
            y_pred = self.train_on_batch(X, y)
            y = y.cpu().numpy()
            
            acc = accuracy_score(y, np.argmax(y_pred, axis=1))
            loss = log_loss(y, y_pred, labels=range(N_CLASSES))
            
            mean_acc = self.ewa(acc, mean_acc, 0.1)
            mean_loss = self.ewa(loss, mean_loss, 0.1)
            
            elapsed_time = round(time.time() - start_time)
            elapsed_time = datetime.timedelta(seconds=elapsed_time)
            print("[{0}] Train step {1}/{2}\tLoss: {3:.6f}\tAccuracy: {4:.6f}".format(
                elapsed_time,
                batch_num + 1,
                batches_per_epoch,
                mean_loss,
                mean_acc
            ))
    
    
    def predict(self, data_loader):
        self.set_predict_mode()
        preds = []
        batches_per_dl = len(data_loader)
        start_time = time.time()

        for batch_num, X in enumerate(data_loader):
            if type(X) == list:
                X = X[0]

            y_pred = self.predict_on_batch(X)
            preds.append(y_pred)
            elapsed_time = round(time.time() - start_time)
            elapsed_time = datetime.timedelta(seconds=elapsed_time)
            print("[{0}] Predict step {1}/{2}\t ".format(
                elapsed_time, batch_num, batches_per_dl))
        
        return np.concatenate(preds)
    
    
    def save(self, fn):
        self.model.save_model(fn)
    
    
    def load(self, fn):
        self.model.load_model(fn)