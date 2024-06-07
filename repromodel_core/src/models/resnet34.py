# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class ResNet34(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 1000, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the ResNet34 model with the specified parameters
        self.resnet = models.resnet34(pretrained=self.pretrained)
        
        # Modify the final fully connected layer to have the specified number of classes
        if self.num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        return self.resnet(x)

class _TestResNet34(unittest.TestCase):
    def test_resnet34_initialization(self):
        # Test with default parameters
        model = ResNet34()
        self.assertIsInstance(model, ResNet34, "Model is not an instance of ResNet34")
        self.assertEqual(model.num_classes, 1000, "Default num_classes is not 1000")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = ResNet34(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = ResNet34(num_classes=1000, pretrained=True)
        self.assertEqual(model.num_classes, 1000, "Pretrained num_classes should be 1000")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_resnet34_forward_pass(self):
        model = ResNet34(num_classes=10, pretrained=False)
        input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10), f"Output shape is not correct: {output.shape}")

    def test_resnet34_tags(self):
        # Check if the class has the correct tags
        model = ResNet34()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()