import os
import datetime
import argparse
import multiprocessing
from clickclick.cv_trainer import train_fold
from clickclick.utils import save_full_train, save_kfold_data


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='data')
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--extra_dir', default=None,
                        help='Directory to extra training data')
    parser.add_argument('--full_train_name', default='full_train.csv',
                        help='File name to save full train DataFrame to `src_dir`')
    parser.add_argument('--cv_dir', required=True,
                        help='Directory to store train/val splits')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    for arg in vars(args):
        print("{0}: {1}".format(arg, getattr(args, arg)))

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    
    cv_path = os.path.join(args.src_dir, args.cv_dir)
    # if the path for cv data passed does not exist then make it and store 
    # train/validation splits in it
    if not os.path.exists(cv_path):
        save_full_train(src_dir=args.src_dir, train_dir=args.train_dir, 
                        filename_out=args.full_train_df,
                        extra_dir=args.extra_dir)
        save_kfold_data(args.src_dir, args.full_train_df, args.cv_dir,
                        k=args.num_folds)
    
    for fold_id in range(1, args.num_folds+1):
        print("\nStart training fold {0}/{1} @ {2}".format(fold_id,
            args.num_folds, datetime.datetime.now().strftime("%H:%M:%S")))
        train_fold(cv_data_path=cv_path, fold_id=fold_id, 
            epochs=args.num_epochs,  batch_size=args.batch_size,
            num_workers=args.num_workers, model_dir=args.model_dir)

    print("Finished script @ {}".format(datetime.datetime.now().strftime("%H:%M:%S")))


if __name__ == '__main__':
    main()