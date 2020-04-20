from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse, os, ast
from models import colour_net
from utils import get_training_data, get_test_data, dir_path
from callbacks import LearnedAccuracyWriter
import csv

COLOURS = sorted(["Blue", "Brown",  "Green", "Red", "Grey",  "Purple", "Yellow", "Orange"])
IMG_ROWS, IMG_COLS = 224, 224
NUM_CLASSES = len(COLOURS)
MAX_EPOCHS = 500
PATIENCE = 100
STEPS_PER_EPOCH = 10


def run_experiment(num_models, training_dir, test_dir, results_dir, transform, weights_file=None):
    training_set = get_training_data(training_dir, COLOURS, IMG_ROWS, IMG_COLS, transform=transform)
    test_set = get_test_data(test_dir, COLOURS, IMG_ROWS, IMG_COLS, transform=transform)
    for epoch in range(num_models):
        callbacks_list = []
        if weights_file:
            callbacks_list.append(ModelCheckpoint(weights_file, monitor='val_accuracy', verbose=1, save_best_only=True))
        callbacks_list.append(LearnedAccuracyWriter(COLOURS, test_set, epoch, PATIENCE, results_dir))
        callbacks_list.append(EarlyStopping(monitor='val_accuracy', patience=PATIENCE))
        model = colour_net(NUM_CLASSES, weights_file)
        model.fit_generator(
            training_set,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=MAX_EPOCHS,
            validation_data=test_set,
            callbacks=callbacks_list,
            verbose=1)


def process_results(result_dir):
    file_list = []
    result_list = []
    epochs_list = []
    for (dirpath, dirnames, filenames) in os.walk(result_dir):
        file_list += [os.path.join(dirpath, file) for file in filenames
                        if '.txt' in file]

    for file_path in file_list:
        with open(os.path.join(file_path), 'r') as fh:
            result_list.append(ast.literal_eval(fh.readline()))
            epochs_list.append(int(ast.literal_eval(fh.readline())))

    print('Total epochs', len(epochs_list))
    keys = list(result_list[0].keys())
    length = float(len(result_list))
    for key in keys:
        print('{}={}'.format(key, sum([res[key] for res in result_list])/ length))
    print('Average', sum(epochs_list)/ float(len(epochs_list)))
    print('Min', min(epochs_list))
    print('Max', max(epochs_list))

    with open('{}.csv'.format(result_dir.replace('/', '')), 'w', newline='') as file_name:
        writer = csv.writer(file_name)
        writer.writerow(sorted(list(result_list[0].keys())) + ['total'])
        for i in range(len(result_list)):
            writer.writerow([result_list[i][key] for key in sorted(list(result_list[i].keys()))] + [epochs_list[i]])

    with open('{}.csv'.format(result_dir.replace('/', '')), 'w', newline='') as file_name:
        writer = csv.writer(file_name)
        writer.writerow(sorted(list(result_list[0].keys())) + ['total'])
        for i in range(len(result_list)):
            writer.writerow([result_list[i][key] for key in sorted(list(result_list[i].keys()))] + [epochs_list[i]])
    
    answer = []
    reader = csv.DictReader(open('{}.csv'.format(result_dir.replace('/', ''))))
    for i in list(reader)[:-1]:                    
        for key, val in i.items():                             
            answer.append((key, val)) 

    with open('{}_Rprocessible.csv'.format(result_dir.replace('/', '')), 'w') as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["colour", "epoch"])
        for i in answer:
            if 'total' in str(i):
                continue
            writer.writerow([i[0], i[1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run_experiment', action='store_true', help='Execute pilot study 1')
    group.add_argument('--process_results', action='store_true', help='Process results of pilot study 1')
    parser.add_argument('--weight_file', type=str, help='Load/store weights in file')
    parser.add_argument('--train_dir', type=str, help='Directory of training data', default='data/training_data')
    parser.add_argument('--test_dir', type=str, help='Directory of test data', default='data/val_data')
    parser.add_argument('--num_models', type=int, help='Number of models to create', default=500)
    parser.add_argument('--results_dir', type=dir_path, help='Directory of results from pilot study 1', default='results')
    parser.add_argument('--transform', type=str, help='Directory of results from pilot study 1', default='rgb')
    args = parser.parse_args()
    if args.run_experiment:
        run_experiment(args.num_models, args.train_dir, args.test_dir, args.results_dir, args.transform.lower())
    elif args.process_results:
        process_results(args.results_dir)
