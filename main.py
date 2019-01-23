from train import train_fold
from utils import create_kfold_data
import datetime
import multiprocessing

SRC_DIR = 'data'
TRAIN_FN = 'train'
MODE = 'train'
NUM_FOLDS = 5
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = multiprocessing.cpu_count()


if __name__ == '__main__':
	if MODE == 'train':
		create_kfold_data(SRC_DIR, TRAIN_FN, NUM_FOLDS)
		for fold_id in range(1, NUM_FOLDS+1):
			print("Start training fold {0} / {1} @ {2}".format(fold_id,
				NUM_FOLDS, datetime.datetime.now().strftime("%H:%M:%S"))
			train_fold(fold_id, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
				       n_workers=NUM_WORKERS)
	else:
		pass