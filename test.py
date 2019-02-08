import pickle
import numpy as np
import os
import argparse
import datetime
import pandas as pd
from clickclick.utils import prep_test_data
from clickclick.augmentation import CameraAugmentor
from clickclick.dataset import CameraDataset
from clickclick.model import CameraArchitecture, CameraModel
from clickclick.postprocessing import generate_submission
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import multiprocessing


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--src_dir', default='data')
	parser.add_argument('--test_dir', default='test')
	parser.add_argument('--full_test_name', default='test.csv')
	parser.add_argument('--model_dir', required=True)
	parser.add_argument('--submission_path', required=True)
	parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
	parser.add_argument('--batch_size', type=int, default=64)
	

	args = parser.parse_args()

	full_test_path = os.path.join(args.src_dir, args.full_test_name)
	if not os.path.exists(full_test_path):
		prep_test_data(args.src_dir, args.test_dir, full_test_path)
	
	test_aug = CameraAugmentor(train_mode=False)
	test_ds = CameraDataset(full_test_path, test_aug, extend_dataset=True)
	test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, 
	                     num_workers=args.num_workers)

	model_list = [os.path.join(args.model_dir, fn) for fn in os.listdir(args.model_dir)]
	n_models = len(model_list)

	base_model = CameraArchitecture(resnet50)
	camera_model = CameraModel(base_model)
	pred_list = []

	for i, model_path in enumerate(model_list[:2]):
		print("\n[{0}/{1}] Loading {2} for test prediction".format(
			i+1, n_models, model_path))
		camera_model.load(model_path)
		preds = camera_model.predict(test_dl)
		pred_list.append(preds)

	test_filenames = test_ds.get_filenames()
	generate_submission(pred_list, filenames=test_filenames, 
						output_path=args.submission_path)

	print("Finished script @ {}".format(datetime.datetime.now().strftime("%H:%M:%S")))


if __name__ == '__main__':
	main()