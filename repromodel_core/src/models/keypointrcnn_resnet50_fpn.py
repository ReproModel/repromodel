import torch
import torch.nn as nn
import torchvision.models.detection as models
import unittest

from ..decorators import enforce_types_and_ranges, tag

#standardize the output for torchvision models
from ..utils import extract_output

@tag(task=["object detection", "instance segmentation", "keypoint detection"], 
     subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class KeypointRCNNResNet50FPN(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 2, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False},
        'num_keypoints': {'type': int, 'default': 17, 'range': (1, 100)}
    })
    def __init__(self, num_classes=2, pretrained=False, num_keypoints=17):
        super(KeypointRCNNResNet50FPN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_keypoints = num_keypoints

        self.keypointrcnn = models.keypointrcnn_resnet50_fpn(pretrained=self.pretrained, num_classes=2)
        if self.num_classes != 2 or self.num_keypoints != 17:
            in_features = self.keypointrcnn.roi_heads.box_predictor.cls_score.in_features
            self.keypointrcnn.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

            keypoint_predictor_in_channels = self.keypointrcnn.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
            self.keypointrcnn.roi_heads.keypoint_predictor = models.keypoint_rcnn.KeypointRCNNPredictor(keypoint_predictor_in_channels, self.num_keypoints)

    def forward(self, x, targets=None):
        return extract_output(self.keypointrcnn(x, targets), model_type='keypointrcnn')

class _TestKeypointRCNNResNet50FPN(unittest.TestCase):
    def test_keypointrcnn_resnet50_fpn_initialization(self):
        model = KeypointRCNNResNet50FPN()
        self.assertIsInstance(model, KeypointRCNNResNet50FPN, "Model is not an instance of KeypointRCNNResNet50FPN")
        self.assertEqual(model.num_classes, 2, "Default num_classes is not 2")
        self.assertEqual(model.num_keypoints, 17, "Default num_keypoints is not 17")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        model = KeypointRCNNResNet50FPN(num_classes=10, pretrained=False, num_keypoints=20)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertEqual(model.num_keypoints, 20, "Custom num_keypoints is not 20")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        model = KeypointRCNNResNet50FPN(num_classes=10, pretrained=True, num_keypoints=20)
        self.assertEqual(model.num_classes, 10, "Pretrained num_classes should be 10")
        self.assertEqual(model.num_keypoints, 20, "Pretrained num_keypoints should be 20")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_keypointrcnn_resnet50_fpn_forward_pass(self):
        model = KeypointRCNNResNet50FPN(num_classes=10, pretrained=False, num_keypoints=20)
        model.eval()  # Ensure the model is in evaluation mode
        input_tensor = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]  # Example input tensor for KeypointRCNNResNet50FPN
        output = model(input_tensor)
        self.assertTrue(isinstance(output, torch.Tensor), "Output should be type torch.Tensor")

    def test_keypointrcnn_resnet50_fpn_tags(self):
        model = KeypointRCNNResNet50FPN()
        self.assertEqual(model.task, ["object detection", "instance segmentation", "keypoint detection"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()