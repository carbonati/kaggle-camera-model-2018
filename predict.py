import os
import argparse
import datetime
from utils import prep_test_data
from augmentation import CameraAugmentor
from dataset import CameraDataset
from torch.utils.data import DataLoader
from models import BuildModel, CameraModel
from torchvision.models import resnet50, resnet34


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', required=True)
	parser.add_argument('test_df_path', default='data/test.csv')
	parser.add_argument('src_dir', default='data')
	parser.add_argument('test_dir', default='test')

	args = parser.parse_args()

	if not os.path.exists(args.test_df_path):
		prep_test_data(src_dir, test_dir)

	test_aug = CameraAugmentor(train_mode=False)
	test_ds = CameraDataset(args.test_df_path, test_aug, extend_dataset=True)
	test_dl = DataLoader(test_ds, batch_size=2, shuffle=False, drop_last=False, 
	                     num_workers=4)

	model_ls = [os.path.join(mode_dir, fn) for fn in os.listdir(args.model_dir)]
	n_models = len(model_ls)

	base_model = BuildModel(resnet50)
	camera_model = CameraModel(base_model)
	pred_ls = []

	for i, model_fp in model_ls:
		print("[{0}/{1}] Loading model {2} for test prediction".format(
			i+1, n_models, model_fp))
		camera_model.load(model_fp)
		preds = camera_model.predict(test_dl)
		pred_ls.append(preds)

	## call some postprocessing function

if __name__ == '__main__':
	main()