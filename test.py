import ltr.models.depth as depth
from torchsummary import summary
import torch
from ltr.dataset import CDTB
from ltr.data.sampler import DepthSampler 
import ltr.data.transforms as tfm
from ltr.data import processing

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

search_area_factor = 5.0
feature_sz = 18
output_sz = feature_sz * 16
center_jitter_factor = {'train': 0, 'test': 4.5}
scale_jitter_factor = {'train': 0, 'test': 0.5}

# The joint augmentation transform, that is applied to the pairs jointly
transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

# The augmentation transform applied to the training set (individually to each image in the pair)
transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                tfm.Normalize(mean=normalize_mean, std=normalize_std))

proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
data_processing_train = processing.DepthProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_sz,
                                                    center_jitter_factor=center_jitter_factor,
                                                    scale_jitter_factor=scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    transform=transform_train,
                                                    joint_transform=transform_joint)

x = CDTB()

dataset_train = DepthSampler([x], p_datasets=None, samples_per_epoch=1000*64, max_gap=50, num_test_frames=1, num_train_frames=1, processing=data_processing_train)
dataset_train.__getitem__(0)

print('--------------')