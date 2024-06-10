# wrapped model from torchvision library. Corresponding License stated in the ReproModel repository.
import torch
import torch.nn as nn
import torchvision.models as models
import unittest

# Assuming the enforce_types_and_ranges and tag decorators are defined in decorators.py
from ..decorators import enforce_types_and_ranges, tag  # Adjust the import path accordingly

@tag(task=["classification"], subtask=["binary", "multi-class"], modality=["images"], submodality=["RGB"])
class GoogLeNet(nn.Module):
    @enforce_types_and_ranges({
        'num_classes': {'type': int, 'default': 1000, 'range': (1, 10000)},
        'pretrained': {'type': bool, 'default': False},
        'aux_logits': {'type': bool, 'default': True},
        'transform_input': {'type': bool, 'default': False}
    })
    def __init__(self, num_classes=1000, pretrained=False, aux_logits=True, transform_input=False):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # Load the GoogLeNet model with the specified parameters
        self.googlenet = models.googlenet(pretrained=self.pretrained, aux_logits=self.aux_logits, transform_input=self.transform_input)
        
        # Modify the final classifier layer to have the specified number of classes
        if self.num_classes != 1000:
            self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, self.num_classes)
            if self.aux_logits:
                self.googlenet.aux1 = models.inception.InceptionAux(512, self.num_classes)
                self.googlenet.aux2 = models.inception.InceptionAux(528, self.num_classes)

    def forward(self, x):
        return self.googlenet(x)

class _TestGoogLeNet(unittest.TestCase):
    def test_googlenet_initialization(self):
        # Test with default parameters
        model = GoogLeNet()
        self.assertIsInstance(model, GoogLeNet, "Model is not an instance of GoogLeNet")
        self.assertEqual(model.num_classes, 1000, "Default num_classes is not 1000")
        self.assertFalse(model.pretrained, "Default pretrained is not False")
        self.assertTrue(model.aux_logits, "Default aux_logits is not True")
        self.assertFalse(model.transform_input, "Default transform_input is not False")

        # Test with custom parameters
        model = GoogLeNet(num_classes=10, pretrained=False, aux_logits=False, transform_input=True)
        self.assertEqual(model.num_classes, 10, "Custom num_classes is not 10")
        self.assertFalse(model.pretrained, "Custom pretrained is not False")
        self.assertFalse(model.aux_logits, "Custom aux_logits is not False")
        self.assertTrue(model.transform_input, "Custom transform_input is not True")

        # Test with pretrained model
        model = GoogLeNet(num_classes=1000, pretrained=True)
        self.assertEqual(model.num_classes, 1000, "Pretrained num_classes should be 1000")
        self.assertTrue(model.pretrained, "Pretrained parameter should be True")

    def test_googlenet_forward_pass(self):
        model = GoogLeNet(num_classes=10, pretrained=False)
        input_tensor = torch.randn(2, 3, 299, 299)  # Updated input tensor for GoogLeNet with batch size 2
        output = model(input_tensor)
        self.assertEqual(output[0].shape, (2, 10), f"Output shape is not correct: {output[0].shape}")

    def test_googlenet_tags(self):
        # Check if the class has the correct tags
        model = GoogLeNet()
        self.assertEqual(model.task, ["classification"], "Task tag is incorrect")
        self.assertEqual(model.subtask, ["binary", "multi-class"], "Subtask tag is incorrect")
        self.assertEqual(model.modality, ["images"], "Modality tag is incorrect")
        self.assertEqual(model.submodality, ["RGB"], "Submodality tag is incorrect")

if __name__ == "__main__":
    unittest.main()