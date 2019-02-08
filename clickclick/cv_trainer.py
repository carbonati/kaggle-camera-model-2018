import os
import datetime
from dataset import CameraDataset
from augmentation import CameraAugmentor
from torch.utils.data import DataLoader
from model import CameraArchitecture, CameraModel
from torchvision.models import resnet50, resnet34
from sklearn.metrics import log_loss, accuracy_score


def train_fold(cv_data_path, fold_id, epochs, batch_size, num_workers, model_dir='models'):
    train_path = os.path.join(cv_data_path, 'train_{}.csv'.format(fold_id))
    val_path = os.path.join(cv_data_path, 'val_{}.csv'.format(fold_id))

    train_aug = CameraAugmentor(train_mode=True)
    val_aug = CameraAugmentor(train_mode=False)

    train_ds = CameraDataset(train_path, train_aug)
    val_ds = CameraDataset(val_path, val_aug)
    y_val = val_ds.get_labels()

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = CameraArchitecture(resnet50)
    camera_model = CameraModel(model)

    for epoch in range(epochs):
        camera_model.fit(train_dl)
        y_pred = camera_model.predict(val_dl)

        val_loss = log_loss(y_val, y_pred)
        val_acc = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))

        camera_model.scheduler_step(val_loss, epoch)
        lr = camera_model.optimizer.param_groups[0]['lr']
        print("\nEpoch {0}: Val Accuracy {1:.6f}\tVal Loss {2:.6f}\tlr {3}\n".format(
            epoch+1, val_acc, val_loss, lr))

    model_name = 'camera_model_{0}.pth'.format(fold_id)
    model_path = os.path.join(model_dir, model_name)
    print("Saving model to {0} @ {1}\n".format(model_path, 
        datetime.datetime.now().strftime("%H:%M:%S")))
    camera_model.save(model_path)