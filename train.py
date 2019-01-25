import os
import datetime
import argparse
import multiprocessing
from cv_trainer import train_fold
from utils import create_kfold_data


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--src_dir', default='data')
	parser.add_argument('--train_dir', default='train')
	parser.add_argument('--model_dir', default='models_{0}'.format(
		datetime.datetime.now().strftime("%Y_%m_%d")))
	parser.add_argument('--num_folds', type=int, default=5)
	parser.add_argument('--num_epochs', type=int, default=80)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())

	args = parser.parse_args()
	for arg in vars(args):
		print("{0} = {1}".format(arg, getattr(args, arg)))

	if not os.path.exists(args.model_dir):
		os.mkdir(args.model_dir)
	create_kfold_data(args.src_dir, args.train_dir, args.num_folds)
	
	for fold_id in range(1, args.num_folds+1):
		print("\nStart training fold {0}/{1} @ {2}".format(fold_id,
			args.num_folds, datetime.datetime.now().strftime("%H:%M:%S")))
		train_fold(fold_id, epochs=args.num_epochs, batch_size=args.batch_size, 
			       n_workers=args.num_workers, model_dir=args.model_dir)	


if __name__ == '__main__':
	main()