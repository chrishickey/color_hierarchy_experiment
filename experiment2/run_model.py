from engine import train_one_epoch, evaluate
import utils, torch, sys
import transforms as T
from data_loader_coloured import CustomColorAnnotatedDataloader, INT_CLOTHING_CATEGORY
from custom_transforms import BgrToOpp
from models import get_model
from custom_evaluator2 import custom_evaluate
import subprocess, argparse
NUM_TRIALS = 100
NUM_EPOCHS = 100


def main(color_space, img_dir, train_dir, test_dir, identifier):
    # Define Transforms
    #subprocess.check_output('rm -f ./mAP/input/*/*'.split(' '))
    if color_space == "OPP":
        train_transforms = [BgrToOpp(), T.ToTensor(), T.RandomHorizontalFlip(0.5)]
        test_transforms = [BgrToOpp(), T.ToTensor()]
    else:
        train_transforms = [T.ToTensor(), T.RandomHorizontalFlip(0.5)]
        test_transforms = [T.ToTensor()]

    train_transforms = T.Compose(train_transforms)
    test_transforms = T.Compose(test_transforms)

    dataset = CustomColorAnnotatedDataloader(train_dir, img_dir, train_transforms, color_space=color_space)
    dataset_test = CustomColorAnnotatedDataloader(test_dir, img_dir, test_transforms, color_space=color_space)
    
    dataset.ordered_files = dataset.ordered_files
    dataset_test.ordered_files = dataset_test.ordered_files

    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=8, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                    batch_size=8, shuffle=False, num_workers=4,collate_fn=utils.collate_fn)
    
    colours = dataset_test.colours
    identifier = "{}_{}".format(color_space, identifier)
    for i in range(NUM_TRIALS):

        model, optimizer, lr_scheduler = get_model(INT_CLOTHING_CATEGORY, scheduler=None)
        #model.roi_heads.box_predictor = FastRCNNPredictor(1024, len(categories))
        device = torch.device('cuda')
        model.to(device)
        
        for epoch in range(NUM_EPOCHS):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500)
            
            accuracy, percentage_dict = custom_evaluate(model, data_loader_test, INT_CLOTHING_CATEGORY, colours, device,
                                                        "{}_{}.txt".format(identifier, i), identifier="{}_epoch{}".format(identifier, epoch))

            print('Accuracy is {}'.format(accuracy))
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(identifier, epoch, accuracy))

            if list(percentage_dict.values()) and all([i >= 0.75 for i in percentage_dict.values()]):
                print('All accuracy over .75 threshold')
                break

            if lr_scheduler:
                print('Incrementing scheduler')
                lr_scheduler.step()

        del model
        del lr_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute experiment2 for colored object detection..')
    parser.add_argument('--color_space', type=str, help='Color space to use in experiment (opp, rgb, yuv, ybr)')
    parser.add_argument('--images_dir', type=str, help='Directory where images are stored')
    parser.add_argument('--train_annotations', type=str, help='Color annotations for training data')
    parser.add_argument('--test_annotations', type=str, help='Color annotations for test data')
    parser.add_argument('--identifier', type=str, help='Identifiet to recognise files created through')
    args = parser.parse_args()
    main(args.color_space.upper(), args.images_dir, args.train_annotations, args.test_annotations, args.identifier)
