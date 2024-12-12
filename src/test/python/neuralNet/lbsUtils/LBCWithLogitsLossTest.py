import unittest
from unittest.mock import patch, MagicMock
import torch
from src.main.python.neuralNet.lbcUtils.LBCWithLogitsLoss import LBCWithLogitsLoss

class TestLBCWithLogitsLoss(unittest.TestCase):
    def setUp(self):
        """Set up common test variables and mocks."""
        # Patch the logger to avoid real logging during tests
        self.logger_patcher = patch('src.main.resources.CreateLogger')
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        # Sample data for tests
        self.n_categories_total = 10
        self.subset = [0, 2, 4]
        self.device_cpu = 'cpu'
        self.device_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Logits and targets for testing
        self.logits = torch.tensor([[1.2, -0.8, 0.3]], dtype=torch.float32)
        self.targets = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    def tearDown(self):
        """Stop patching the logger."""
        self.logger_patcher.stop()

    def test_initialization_cpu(self):
        """Test the initialization on the CPU device."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cpu)
        print(self.mock_logger.mock_calls)  # Debugging line to see all logged calls
        self.mock_logger.info.assert_any_call("Initializing starts")
        self.mock_logger.info.assert_any_call("Initializing ends")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this system.")
    def test_initialization_cuda(self):
        """Test the initialization on the CUDA device."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cuda)
        self.assertEqual(loss_fn.device, 'cuda')
        self.assertTrue(loss_fn.pos_weight.device.type == 'cuda')
        self.mock_logger.info.assert_any_call("Initializing starts")
        self.mock_logger.info.assert_any_call("Initializing ends")

    def test_confusion_weight(self):
        """Test the confusion_weight method for correct weight calculation."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cpu)
        expected_weights = torch.tensor([0.1, 0.3, 0.5], device=self.device_cpu)
        self.assertTrue(torch.allclose(loss_fn.pos_weight, expected_weights))
        self.mock_logger.info.assert_any_call("confusion_weight starts")
        self.mock_logger.info.assert_any_call("confusion_weight ends")

    def test_confusion_weight_invalid_subset(self):
        """Test the confusion_weight method with an invalid subset."""
        invalid_subset = [0, 11]  # Index 11 is out of bounds
        with self.assertRaises(ValueError) as context:
            LBCWithLogitsLoss(self.n_categories_total, invalid_subset, device=self.device_cpu)
        self.assertIn("Failed to calculate confusion weights", str(context.exception))
        self.mock_logger.error.assert_called_once()

    def test_call_loss_computation(self):
        """Test the __call__ method for correct loss computation."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cpu)
        loss = loss_fn(self.logits, self.targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertAlmostEqual(loss.item(), 0.3165, places=4)
        self.mock_logger.info.assert_any_call("__call__ starts")
        self.mock_logger.info.assert_any_call("__call__ ends")

    def test_call_with_exception_handling(self):
        """Test exception handling during the __call__ method."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cpu)
        with patch('torch.nn.LogSigmoid.__call__', side_effect=RuntimeError("LogSigmoid failed")):
            with self.assertRaises(RuntimeError) as context:
                loss_fn(self.logits, self.targets)
            self.assertIn("Failed to compute the loss", str(context.exception))
            self.mock_logger.error.assert_called_once_with("Error during loss computation: LogSigmoid failed")

    def test_call_with_invalid_input_shapes(self):
        """Test __call__ method with mismatched logits and targets shapes."""
        loss_fn = LBCWithLogitsLoss(self.n_categories_total, self.subset, device=self.device_cpu)
        invalid_targets = torch.tensor([[1, 0]], dtype=torch.float32)  # Shape mismatch
        with self.assertRaises(RuntimeError) as context:
            loss_fn(self.logits, invalid_targets)
        self.assertIn("Failed to compute the loss", str(context.exception))
        self.mock_logger.error.assert_called_once()

if __name__ == "__main__":
    unittest.main()
