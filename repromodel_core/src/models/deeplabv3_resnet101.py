# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from repromodel_core.src.decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

#standardize the output for torchvision models
from ..utils import extract_output

@tag(task=["segmentation"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class DeepLabV3ResNet101(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 21, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=21, pretrained=False):
        super(DeepLabV3ResNet101, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the DeepLabV3 ResNet101 model with the specified parameters
        self.deeplabv3 = models.deeplabv3_resnet101(pretrained=self.pretrained)

        # Modify the classifier layer to have the specified number of classes
        if num_classes != 21:
            self.deeplabv3.classifier[-1] = nn.Conv2d(self.deeplabv3.classifier[-1].in_channels, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        return extract_output(self.deeplabv3(x))

class _TestDeepLabV3ResNet101(unittest.TestCase):
    def test_deeplabv3_resnet101_initialization(self):
        # Test with default parameters
        model = DeepLabV3ResNet101()
        self.assertIsInstance(model, DeepLabV3ResNet101, "Model is not an instance of DeepLabV3ResNet101")
        self.assertEqual(model.num_classes, 21, "Default num_classes is not 21")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = DeepLabV3ResNet101(num_classes=10, pretrained=True)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertTrue(model.pretrained, "Custom pretrained is not True")

        # Test with pretrained model
        model = DeepLabV3ResNet101(num_classes=21, pretrained=True)
        self.assertEqual(model.num_classes, 21, "Pretrained num_classes should be 21")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_deeplabv3_resnet101_forward_pass(self):
        model = DeepLabV3ResNet101(num_classes=10, pretrained=False)
        input_tensor = torch.randn(2, 3, 224, 224)  # Example input tensor for DeepLabV3ResNet101 with batch size 2
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 10, 224, 224), f"Output shape is not correct: {output.shape}")

    def test_deeplabv3_resnet101_tags(self):
        # Check if the class has the correct tags
        model = DeepLabV3ResNet101()
        self.assertEqual(model.task, ["segmentation"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()