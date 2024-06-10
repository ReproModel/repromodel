# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.

import torch
import torch.nn as nn
import torchvision.models as models
import unittest
from ..decorators import enforce_types_and_ranges, tag

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class AlexNet(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 1000, 'range': (1, 10000)},
        'dropout': {'type': float, 'default': 0.5, 'range': (0.0, 1.0)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=1000, dropout=0.5, pretrained=False):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.pretrained = pretrained

        # Load the AlexNet model
        self.alexnet = models.alexnet(pretrained=self.pretrained)

        # Modify the classifier to use the specified number of classes and dropout rate
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        return self.alexnet(x)
    
class _TestAlexNet(unittest.TestCase):
    def test_alexnet_initialization(self):
        # Test with default parameters
        model = AlexNet()
        self.assertIsInstance(model, AlexNet, "Model is not an instance of AlexNet")
        self.assertEqual(model.num_classes, 1000, "Default num_classes is not 1000")
        self.assertEqual(model.dropout, 0.5, "Default dropout is not 0.5")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = AlexNet(num_classes=10, dropout=0.3, pretrained=True)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertEqual(model.dropout, 0.3, "Custom dropout is not 0.3")
        self.assertTrue(model.pretrained, "Custom pretrained is not True")

    def test_alexnet_forward_pass(self):
        model = AlexNet(num_classes=10)
        input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10), f"Output shape is not correct: {output.shape}")

    def test_alexnet_tags(self):
        # Check if the class has the correct tags
        model = AlexNet()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()