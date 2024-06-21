# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models.detection as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

#standardize the output for torchvision models
from ..utils import extract_output

@tag(task=["object detection", "instance segmentation", "keypoint detection"], 
     subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class RetinaNetResNet50FPN(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 91, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=91, pretrained=False):
        super(RetinaNetResNet50FPN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the RetinaNet ResNet50 FPN model with the specified parameters
        self.retinanet = models.retinanet_resnet50_fpn(pretrained=self.pretrained, num_classes=self.num_classes)

    def forward(self, x, targets=None):
        return extract_output(self.retinanet(x, targets))

class _TestRetinaNetResNet50FPN(unittest.TestCase):
    def test_retinanet_resnet50_fpn_initialization(self):
        # Test with default parameters
        model = RetinaNetResNet50FPN()
        self.assertIsInstance(model, RetinaNetResNet50FPN, "Model is not an instance of RetinaNetResNet50FPN")
        self.assertEqual(model.num_classes, 91, "Default num_classes is not 91")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = RetinaNetResNet50FPN(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = RetinaNetResNet50FPN(num_classes=91, pretrained=True)
        self.assertEqual(model.num_classes, 91, "Pretrained num_classes should be 91")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_retinanet_resnet50_fpn_forward_pass(self):
        model = RetinaNetResNet50FPN(num_classes=10, pretrained=False)
        model.eval()  # Ensure the model is in evaluation mode
        input_tensor = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]  # Example input tensor for RetinaNetResNet50FPN
        output = model(input_tensor)
        self.assertTrue(isinstance(output, list) and isinstance(output[0], dict), "Output is not a list of dicts")

    def test_retinanet_resnet50_fpn_tags(self):
        # Check if the class has the correct tags
        model = RetinaNetResNet50FPN()
        self.assertEqual(model.task, ["object detection", "instance segmentation", "keypoint detection"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()