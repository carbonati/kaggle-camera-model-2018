import os
import datetime
from dataset import CameraDataset
from augmentation import CameraAugmentor
from torch.utils.data import DataLoader
from model import BuildModel, CameraModel
from torchvision.models import resnet50, resnet34
from sklearn.metrics import log_loss, accuracy_score


def train_fold(fold_id, epochs=1, batch_size=64, n_workers=1, src_dir='data'):
    fold_fn = 'fold_{}'.format(fold_id)
    fold_fp = os.path.join(src_dir, fold_fn)

    data_files = os.listdir(fold_fp)
    train_fp = [os.path.join(fold_fp, fn) for fn in data_files if 'train' in fn][0]
    val_fp = [os.path.join(fold_fp, fn) for fn in data_files if 'val' in fn][0]

    train_aug = CameraAugmentor(train_mode=True)
    val_aug = CameraAugmentor(train_mode=False)

    train_ds = CameraDataset(train_fp, train_aug)
    val_ds = CameraDataset(val_fp, val_aug)
    y_val = val_ds.get_labels()

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=n_workers)

    model = BuildModel(resnet50)
    camera_model = CameraModel(model)

    for epoch in range(epochs):
        camera_model.fit(train_dl)
        y_pred = camera_model.predict(val_dl)

        val_loss = log_loss(y_val, y_pred)
        val_acc = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))

        camera_model.scheduler_step(val_loss, epoch)
        lr = camera_model.optimizer.param_groups[0]['lr']
        print("\nEpoch {0}: Val Accuracy {1:.6f}\tVal Loss {2:.6f}\tlr {3}\n".format(
            epoch, val_acc, val_loss, lr))

    model_fn = 'camera_model_{0}.pth'.format(fold_id)
    model_fp = os.path.join(fold_fp, model_fn)
    print("Saving model to {0} @ {1}\n".format(model_fp, 
        datetime.datetime.now().strftime("%H:%M:%S")))
    camera_model.save(model_fp)