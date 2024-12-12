import unittest
from unittest.mock import patch, MagicMock
import torch
from src.main.resources.CreateLogger import CreateLogger
from src.main.python.neuralNet.lbcUtils.LBCLabel import LBCLabel

class TestLBCLabel(unittest.TestCase):
    def setUp(self):
        """Set up common test variables and mocks."""
        # Patch the logging.getLogger method directly
        self.logger_patcher = patch('logging.getLogger')
        self.mock_get_logger = self.logger_patcher.start()

        # Configure the mock logger to have an error method
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

        # Sample data for tests
        self.y_cpu = torch.tensor([1, 3], device='cpu')
        self.subset = [0, 1, 2, 3]

    def tearDown(self):
        """Stop patching the logger."""
        self.logger_patcher.stop()

    def tearDown(self):
        """Stop patching the logger."""
        self.logger_patcher.stop()

    def test_initialization(self):
        """Test that the class initializes correctly."""
        lbclabel = LBCLabel(self.y_cpu, self.subset, device='cpu')
        self.assertTrue(torch.equal(lbclabel.y, self.y_cpu))
        self.assertEqual(lbclabel.subset, self.subset)
        self.assertEqual(lbclabel.device, 'cpu')


    def test_call_with_cpu(self):
        """Test the __call__ method with CPU device."""
        lbclabel = LBCLabel(self.y_cpu, self.subset, device='cpu')
        result = lbclabel()
        expected_result = torch.tensor([[True, False, False, False],
                                        [True, True, True, False]], dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected_result))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this system.")
    def test_call_with_cuda(self):
        """Test the __call__ method with CUDA device."""
        lbclabel = LBCLabel(self.y_cuda, self.subset, device='cuda')
        result = lbclabel()
        expected_result = torch.tensor([[True, False, False, False],
                                        [True, True, True, False]], dtype=torch.bool, device='cuda')
        self.assertTrue(torch.equal(result, expected_result))

    def test_call_with_subset_tensor(self):
        """Test the __call__ method when the subset is provided as a tensor."""
        subset_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        lbclabel = LBCLabel(self.y_cpu, subset_tensor, device='cpu')
        result = lbclabel()
        expected_result = torch.tensor([[True, False, False, False],
                                        [True, True, True, False]], dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected_result))

    def test_call_with_invalid_device(self):
        """Test the __call__ method with an invalid device."""
        with self.assertRaises(RuntimeError):
            lbclabel = LBCLabel(self.y_cpu, self.subset, device='invalid_device')
            lbclabel()

    def test_call_with_empty_subset(self):
        """Test the __call__ method with an empty subset."""
        lbclabel = LBCLabel(self.y_cpu, [], device='cpu')
        result = lbclabel()
        expected_result = torch.empty((2, 0), dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected_result))

    def test_call_with_empty_y(self):
        """Test the __call__ method with an empty y tensor."""
        empty_y = torch.tensor([], dtype=torch.int64)
        lbclabel = LBCLabel(empty_y, self.subset, device='cpu')
        result = lbclabel()
        expected_result = torch.empty((0, 4), dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected_result))


if __name__ == '__main__':
    unittest.main()
