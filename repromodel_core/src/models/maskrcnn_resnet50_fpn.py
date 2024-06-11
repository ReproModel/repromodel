# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

# standardize the output for torchvision models
from ..utils import extract_output

@tag(task=["object detection", "instance segmentation", "keypoint detection"], 
     subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class MaskRCNNResNet50FPN(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 91, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=91, pretrained=False):
        super(MaskRCNNResNet50FPN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the Mask R-CNN ResNet50 FPN model with the specified parameters
        self.maskrcnn = models.maskrcnn_resnet50_fpn(pretrained=self.pretrained)
        
        # Modify the box_predictor layer to have the specified number of classes
        if num_classes != 91:
            in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
            self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # If the model supports masks, modify the mask predictor layer as well
            if self.maskrcnn.roi_heads.mask_predictor is not None:
                in_features_mask = self.maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                self.maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    def forward(self, x, targets=None):
        return extract_output(self.maskrcnn(x, targets), model_type='maskrcnn')

class _TestMaskRCNNResNet50FPN(unittest.TestCase):
    def test_maskrcnn_resnet50_fpn_initialization(self):
        # Test with default parameters
        model = MaskRCNNResNet50FPN()
        self.assertIsInstance(model, MaskRCNNResNet50FPN, "Model is not an instance of MaskRCNNResNet50FPN")
        self.assertEqual(model.num_classes, 91, "Default num_classes is not 91")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = MaskRCNNResNet50FPN(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = MaskRCNNResNet50FPN(num_classes=91, pretrained=True)
        self.assertEqual(model.num_classes, 91, "Pretrained num_classes should be 91")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_maskrcnn_resnet50_fpn_forward_pass(self):
        model = MaskRCNNResNet50FPN(num_classes=10, pretrained=False)
        model.eval()  # Ensure the model is in evaluation mode
        input_tensor = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]  # Example input tensor for MaskRCNNResNet50FPN
        output = model(input_tensor)
        self.assertTrue(isinstance(output, torch.Tensor), "Output should be type torch.Tensor")

    def test_maskrcnn_resnet50_fpn_tags(self):
        # Check if the class has the correct tags
        model = MaskRCNNResNet50FPN()
        self.assertEqual(model.task, ["object detection", "instance segmentation", "keypoint detection"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()