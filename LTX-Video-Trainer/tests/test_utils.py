import unittest
from unittest.mock import patch, MagicMock
import subprocess
import torch
from ltxv_trainer.utils import get_gpu_memory_gb

class TestGetGpuMemoryGb(unittest.TestCase):

    @patch('ltxv_trainer.utils.subprocess.check_output')
    def test_get_gpu_memory_gb_success(self, mock_check_output):
        """Test successful retrieval of GPU memory from nvidia-smi"""
        # Mock output: 4096 MB
        mock_check_output.return_value = "4096"

        device = MagicMock(spec=torch.device)
        device.index = 0

        memory_gb = get_gpu_memory_gb(device)

        self.assertEqual(memory_gb, 4.0)
        mock_check_output.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                "0",
            ],
            encoding="utf-8",
        )

    @patch('ltxv_trainer.utils.subprocess.check_output')
    def test_get_gpu_memory_gb_device_index_none(self, mock_check_output):
        """Test fallback to device index 0 when device.index is None"""
        mock_check_output.return_value = "2048"

        device = MagicMock(spec=torch.device)
        device.index = None

        memory_gb = get_gpu_memory_gb(device)

        self.assertEqual(memory_gb, 2.0)
        mock_check_output.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                "0",
            ],
            encoding="utf-8",
        )

    @patch('ltxv_trainer.utils.torch.cuda.memory_allocated')
    @patch('ltxv_trainer.utils.subprocess.check_output')
    def test_get_gpu_memory_gb_fallback_called_process_error(self, mock_check_output, mock_memory_allocated):
        """Test fallback to torch when nvidia-smi fails with CalledProcessError"""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        # Mock torch memory allocated: 2 GB in bytes
        mock_memory_allocated.return_value = 2 * 1024**3

        device = MagicMock(spec=torch.device)
        device.index = 0

        memory_gb = get_gpu_memory_gb(device)

        self.assertEqual(memory_gb, 2.0)
        mock_memory_allocated.assert_called_once_with(device)

    @patch('ltxv_trainer.utils.torch.cuda.memory_allocated')
    @patch('ltxv_trainer.utils.subprocess.check_output')
    def test_get_gpu_memory_gb_fallback_file_not_found(self, mock_check_output, mock_memory_allocated):
        """Test fallback to torch when nvidia-smi is not found"""
        mock_check_output.side_effect = FileNotFoundError
        # Mock torch memory allocated: 3 GB in bytes
        mock_memory_allocated.return_value = 3 * 1024**3

        device = MagicMock(spec=torch.device)
        device.index = 0

        memory_gb = get_gpu_memory_gb(device)

        self.assertEqual(memory_gb, 3.0)
        mock_memory_allocated.assert_called_once_with(device)

    @patch('ltxv_trainer.utils.torch.cuda.memory_allocated')
    @patch('ltxv_trainer.utils.subprocess.check_output')
    def test_get_gpu_memory_gb_fallback_value_error(self, mock_check_output, mock_memory_allocated):
        """Test fallback to torch when nvidia-smi returns invalid output"""
        mock_check_output.return_value = "invalid output"
        # Mock torch memory allocated: 1 GB in bytes
        mock_memory_allocated.return_value = 1 * 1024**3

        device = MagicMock(spec=torch.device)
        device.index = 0

        memory_gb = get_gpu_memory_gb(device)

        self.assertEqual(memory_gb, 1.0)
        mock_memory_allocated.assert_called_once_with(device)
