import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss
from src.main.python.neuralNet.loops.ConfusionLoop import ConfusionLoop


class TestConfusionLoop(unittest.TestCase):
    def setUp(self):
        """Set up common test variables and mocks."""
        # Create mock model
        self.model = MagicMock()
        self.model.return_value = torch.tensor([[0.6, 0.2], [0.1, 0.9]], dtype=torch.float32)

        # Create a simple dataset: 2 samples, 2 features each
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.int64)
        dataset = TensorDataset(inputs, targets)
        self.dataloader = DataLoader(dataset, batch_size=2)

        # Loss function (mocked for simplicity)
        self.loss_fn = MagicMock()
        self.loss_fn.return_value = torch.tensor(0.5)

        # Other parameters
        self.n_categories = 2
        self.device = 'cpu'
        self.subset = [0, 1]

    def test_initialization(self):
        """Test that ConfusionLoop initializes correctly."""
        with patch.object(LBCWithLogitsLoss, 'confusion_weight', return_value=torch.tensor([0.1, 0.2])):
            confusion_loop = ConfusionLoop(
                dataloader=self.dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                n_categories=self.n_categories,
                device=self.device,
                subset=self.subset
            )

            self.assertEqual(confusion_loop.n_categories, self.n_categories)
            self.assertEqual(confusion_loop.device, self.device)
            self.assertTrue(torch.equal(confusion_loop.torch_weight, torch.tensor([0.1, 0.2])))
            self.assertTrue(torch.equal(confusion_loop.running_conf, torch.zeros(self.n_categories - 1)))
            self.assertEqual(confusion_loop.running_loss, 0)

    def test_call_execution(self):
        """Test the execution of the evaluation loop."""
        with patch.object(LBCWithLogitsLoss, 'confusion_weight', return_value=torch.tensor([0.1, 0.2])):
            confusion_loop = ConfusionLoop(
                dataloader=self.dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                n_categories=self.n_categories,
                device=self.device,
                subset=self.subset
            )

            # Execute the confusion loop
            running_conf, running_loss = confusion_loop()

            # Assertions for running_conf and running_loss
            self.assertTrue(torch.is_tensor(running_conf))
            self.assertAlmostEqual(running_loss, 0.5)

    def test_call_with_empty_dataloader(self):
        """Test the __call__ method with an empty dataloader."""
        empty_dataloader = DataLoader(TensorDataset(), batch_size=2)

        with patch.object(LBCWithLogitsLoss, 'confusion_weight', return_value=torch.tensor([0.1, 0.2])):
            confusion_loop = ConfusionLoop(
                dataloader=empty_dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                n_categories=self.n_categories,
                device=self.device,
                subset=self.subset
            )

            running_conf, running_loss = confusion_loop()

            # Expect zero confusion and zero loss since no data was processed
            self.assertTrue(torch.equal(running_conf, torch.zeros(self.n_categories - 1)))
            self.assertEqual(running_loss, 0)

    def test_confusion_weight_calculation(self):
        """Test the confusion weight calculation during initialization."""
        with patch.object(LBCWithLogitsLoss, 'confusion_weight', return_value=torch.tensor([0.1, 0.3])):
            confusion_loop = ConfusionLoop(
                dataloader=self.dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                n_categories=self.n_categories,
                device=self.device,
                subset=self.subset
            )

            expected_weight = torch.tensor([0.1, 0.3])
            self.assertTrue(torch.equal(confusion_loop.torch_weight, expected_weight))

    def test_call_with_model_exception(self):
        """Test the __call__ method when the model raises an exception."""
        self.model.side_effect = RuntimeError("Model forward pass failed")

        with patch.object(LBCWithLogitsLoss, 'confusion_weight', return_value=torch.tensor([0.1, 0.2])):
            confusion_loop = ConfusionLoop(
                dataloader=self.dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                n_categories=self.n_categories,
                device=self.device,
                subset=self.subset
            )

            with self.assertRaises(RuntimeError) as context:
                confusion_loop()
            self.assertIn("Evaluation loop failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
