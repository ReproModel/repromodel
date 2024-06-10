# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class VGG13(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 1000, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=1000, pretrained=False):
        super(VGG13, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the VGG13 model with the specified parameters
        self.vgg = models.vgg13(pretrained=self.pretrained)

        if self.num_classes != 1000:
            # Modify the classifier to use the specified number of classes
            self.vgg.classifier[6] = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        return self.vgg(x)

class _TestVGG13(unittest.TestCase):
    def test_vgg13_initialization(self):
        # Test with default parameters
        model = VGG13()
        self.assertIsInstance(model, VGG13, "Model is not an instance of VGG13")
        self.assertEqual(model.num_classes, 1000, "Default num_classes is not 1000")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = VGG13(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = VGG13(num_classes=1000, pretrained=True)
        self.assertEqual(model.num_classes, 1000, "Pretrained num_classes should be 1000")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_vgg13_forward_pass(self):
        model = VGG13(num_classes=10, pretrained=False)
        input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10), f"Output shape is not correct: {output.shape}")

    def test_vgg13_tags(self):
        # Check if the class has the correct tags
        model = VGG13()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()