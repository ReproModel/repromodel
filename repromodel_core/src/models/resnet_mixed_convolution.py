import torch
import torch.nn as nn
import torchvision.models.video as video_models
import unittest
from ..decorators import enforce_types_and_ranges, tag

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["video"])
class ResNetMixedConvolution(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 400, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=400, pretrained=False):
        super(ResNetMixedConvolution, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.mc3_18 = video_models.mc3_18(pretrained=self.pretrained, progress=True)
        if self.num_classes != 400:
            in_features = self.mc3_18.fc.in_features
            self.mc3_18.fc = nn.Linear(in_features, self.num_classes)

    def forward(self, x):
        return self.mc3_18(x)

class _TestResNetMixedConvolution(unittest.TestCase):
    def test_resnet_mixed_convolution_initialization(self):
        # Test with default parameters
        model = ResNetMixedConvolution()
        self.assertIsInstance(model, ResNetMixedConvolution, "Model is not an instance of ResNetMixedConvolution")
        self.assertEqual(model.num_classes, 400, "Default num_classes is not 400")
        self.assertFalse(model.pretrained, "Default pretrained is not False")

        # Test with custom parameters
        model = ResNetMixedConvolution(num_classes=10, pretrained=False)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")

        # Test with pretrained model
        model = ResNetMixedConvolution(num_classes=400, pretrained=True)
        self.assertEqual(model.num_classes, 400, "Pretrained num_classes should be 400")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_resnet_mixed_convolution_forward_pass(self):
        model = ResNetMixedConvolution(num_classes=10, pretrained=False)
        input_tensor = torch.randn(1, 3, 8, 112, 112)  # Example input tensor
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10), f"Output shape is not correct: {output.shape}")

    def test_resnet_mixed_convolution_tags(self):
        # Check if the class has the correct tags
        model = ResNetMixedConvolution()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["video"], "Modality tag is incorrect")

if __name__ == "__main__":
    unittest.main()