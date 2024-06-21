# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

#standardize the output for torchvision models
from ..utils import extract_output

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class VGG19_BN(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 1000, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=1000, pretrained=False):
        super(VGG19_BN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the VGG19_BN model with the specified parameters
        self.vgg = models.vgg19_bn(pretrained=self.pretrained)

        if self.num_classes != 1000:
            # Modify the classifier to use the specified number of classes
            self.vgg.classifier[6] = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        return extract_output(self.vgg(x))

class _TestVGG19_BN(unittest.TestCase):
    def test_vgg19_bn_initialization(self):
        # Test with default parameters
        model = VGG19_BN()
        self.assertIsInstance(model, VGG19_BN, "Model is not an instance of VGG19_BN")
        self.assertEqual(model.num_classes, 1000, "Default num_classes is not 1000")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = VGG19_BN(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = VGG19_BN(num_classes=1000, pretrained=True)
        self.assertEqual(model.num_classes, 1000, "Pretrained num_classes should be 1000")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_vgg19_bn_forward_pass(self):
        model = VGG19_BN(num_classes=10, pretrained=False)
        input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10), f"Output shape is not correct: {output.shape}")

    def test_vgg19_bn_tags(self):
        # Check if the class has the correct tags
        model = VGG19_BN()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()