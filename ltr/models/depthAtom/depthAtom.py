import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.depthAtom as depthModels
from ltr import model_constructor
import ltr.models.depth as depth
import torch
from collections import OrderedDict


class DepthATOMnet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, depth_feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(DepthATOMnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.depth_feature_extractor = depth_feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, train_depths, test_imgs, test_depths, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Extract depth features
        train_depths_resize = train_depths.reshape(-1, *train_depths.shape[-2:]).unsqueeze(1)
        test_depths_resize = test_depths.reshape(-1, *test_depths.shape[-2:]).unsqueeze(1)

        train_depth_feat = self.extract_depth_features(train_depths_resize)
        test_depth_feat = self.extract_depth_features(test_depths_resize)

        train_cat = OrderedDict()
        test_cat = OrderedDict()
        for key, value in train_feat.items():
            train_cat[key] = torch.cat((value, train_depth_feat[key]), 1)
        for key, value in test_feat.items():
            test_cat[key] = torch.cat((value, test_depth_feat[key]), 1)

        train_feat_iou = [feat for feat in train_cat.values()]
        test_feat_iou = [feat for feat in test_cat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.reshape(num_train_images, num_sequences, 4),
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def extract_depth_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.depth_feature_extractor(im, layers)

    def extract_depth_nolayer_features(self, im, layers):
        return self.depth_feature_extractor(im, layers)



# @model_constructor
# def atom_resnet18(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
#     # backbone
#     backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

#     # Bounding box regressor
#     iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

#     net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
#                   extractor_grad=False)

#     return net


@model_constructor
def depth_atom_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    
    # depthNet
    depth_net = depth.depthResnet50()

    # Bounding box regressor
    iou_predictor = depthModels.DepthAtomIoUNet(input_dim=(4*256,4*512), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = DepthATOMnet(feature_extractor=backbone_net, depth_feature_extractor=depth_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net
