from copy import deepcopy
import torch
import subprocess
import os
import glob

def get_mAP(targets, predictions, categories, colours, results_file, identifier):
    # Delete ground previous files from run
    files = glob.glob('./mAP_recall/input/ground-truth/*')
    for f in files:
        os.remove(f)

    files = glob.glob('./mAP_recall/input/detection-results/*')
    for f in files:
        os.remove(f)

    # Write the relevant target, prediction information for each file
    # NOTE: while all predictions will be written to the
    for prediction, target in zip(predictions, list(targets)):
        with open("./mAP_recall/input/ground-truth/{}.txt".format(int(target['image_id'])), 'w') as fh:
            for index in range(len(target['boxes'])):
                fh.writelines('{} {} {} {}\n'.format(categories[target['labels'][index]],  ' '.join([str(int(i)) for i in target['boxes'][index]]),  colours[target['colours'][index]], target['certainty'][index]))

        with open("./mAP_recall/input/detection-results/{}.txt".format(int(target['image_id'])), 'w') as fh:
            for index in range(len(prediction['boxes'])):
                fh.writelines('{} {} {}\n'.format(categories[prediction['labels'][index]], round(float(prediction['scores'][index]), 4) , ' '.join([str(int(i)) for i in prediction['boxes'][index]])))

    # This command willcreate the necessary JSON file required to process results
    output = subprocess.check_output(['python', 'mAP_recall/main.py', '-na', '-np', '--set-id', identifier]).decode('utf8')
    print(output)
    #  Extract the mAP VALUE
    output = output.replace('%', '').replace(' AP', '').split('\n')
    a = [i for i in output if i]
    write_dict = { i.replace(' ', '').split('=')[1]: float(i.replace(' ', '').split('=')[0]) / 100 for i in a[:-1]}
    accuracy = float(a[-1].split('=')[1])
    for key, val in write_dict.items():
        print(key, val)

    with open(results_file, "a") as file_object:
        file_object.write("{}\n".format(write_dict))

    with open(results_file, "a") as file_object:
        file_object.write("{}\n".format(accuracy))

    return accuracy, write_dict

def custom_evaluate(model, data_iter, categories, colours, device, results_file, identifier):
    model.eval()
    T = []
    P = []
    with torch.no_grad():
        try:
            for imgs, targets in data_iter:
                predictions = model([img.to(device) for img in imgs])
                T = T + list(deepcopy(targets))
                P = P + predictions
                del imgs
                del targets
        except Exception as e:
            print(str(e))
            raise e
    
    return get_mAP(T, P, categories, colours, results_file, identifier)
