import torch.nn as nn
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import Accuracy, Precision
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.mixture import GaussianMixture 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generator_loss(predictions_fake):
    # hinge loss
    loss = nn.HingeEmbeddingLoss()
    # loss = nn.BCELoss()

    # if batch size = 1, shape raises error
    try: no_predictions = predictions_fake.shape[0]
    except: no_predictions = 1
    
    target = torch.ones(no_predictions).to(device)#.type(torch.LongTensor) # we want generated images to get the label 1 ('real')
    gen_loss = loss(predictions_fake, target) 
    return gen_loss

def discriminator_loss(predictions, target = 'real'):
    loss = nn.HingeEmbeddingLoss()
    # loss = nn.BCELoss()

    # if batch size = 1, shape raises error
    try: no_predictions = predictions.shape[0]
    except: no_predictions = 1

    if target == 'real':
        targets = torch.ones(no_predictions).to(device)#.type(torch.LongTensor) # we want real images to get the label 1 ('real')
    elif target == 'fake':
        targets = torch.zeros(no_predictions).to(device)#.type(torch.LongTensor) # we want generated images to get the label 0 ('fake')
        # targets = -1 * torch.ones(no_predictions).to(device)#.type(torch.LongTensor) # we want generated images to get the label 0 ('fake')

    loss = loss(predictions, targets) 
    return loss

def grid_cell_regularizer(generated_samples, batch_targets):
    """
    Grid cell regularizer.
    ~ Adapted from Deepmind implementation
    """
    weights = torch.ones_like(batch_targets)
    weights[batch_targets == 2] = 2
    loss = torch.mean(torch.abs(generated_samples - batch_targets) * weights)
    return loss

# Evaluations

def SSIM(generated_samples, batch_targets):
    # SSIM requires tensor [batch, channels, width, height]
    generated_samples = generated_samples.unsqueeze(dim=1).to(device)
    batch_targets = batch_targets.unsqueeze(dim=1).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    SSIM_score = ssim(generated_samples, batch_targets)
    return SSIM_score

def classification_metrics(prediction, target):
    """Assign closest class
    Compare one generated image (800,800) to ground truth (800,800)
    """

    flat_x = prediction.view(-1, 1)
    img = flat_x.detach().cpu().numpy()
    gmm = GaussianMixture(n_components=3, means_init=[[0],[1],[2]])
    gmm.fit(img)
    x = gmm.predict(img)
    new_prediction = torch.Tensor(x).view(800,800).to(device)

    acc = Accuracy(task="multiclass", num_classes=4).to(device)
    mac_prec = Precision(task="multiclass", average='macro', num_classes=4).to(device)
    mic_prec = Precision(task="multiclass", average='micro', num_classes=4).to(device)

    accuracy = acc(new_prediction, target)
    macro_precision = mac_prec(new_prediction, target)
    micro_precision = mic_prec(new_prediction, target)

    return accuracy, macro_precision, micro_precision, new_prediction

