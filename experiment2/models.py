
import torchvision, torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
      
def get_resnet_faster_rcnn_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, 
            pretrained_backbone=False)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model(categories, optim_params={'type': 'sgd', 'lr': 0.005}, 
              scheduler={'factor':0.5, 'patience': 2}):

    # our dataset has two classes only - background and person
    num_classes = len(categories)

    # get the model using our helper function
    model = get_resnet_faster_rcnn_model(num_classes)
    # move model to the right device

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if optim_params['type'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=optim_params['lr'])
    elif optim_params['type'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optim_params['lr'],
                                    momentum=0.9, weight_decay=0.0005, nesterov=True)
    else:
        raise Error('Not valid optimizer {}'.format(optim_params['type']))
    

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    if scheduler:
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=scheduler['factor'], patience=scheduler['patience'], verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        #lr_scheduler = None
    else:
        lr_scheduler = None
    
    return model, optimizer, lr_scheduler
        
        
        
        
        
        
        
