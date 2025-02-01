import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import os
from flux_generator import generate_images, check_api_key
import tempfile
import json

class MockInProgress:
    def __init__(self, logs=None):
        self.logs = logs or []

class MockResponse:
    def __init__(self, status=200, content=None):
        self.status = status
        self._content = content or b""
        
    async def read(self):
        return self._content
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, *args):
        pass

class MockClientSession:
    def __init__(self):
        self.response = MockResponse(content=b"fake_image_data")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, *args):
        pass
        
    def get(self, url):
        """Return the response directly since it already implements __aenter__ and __aexit__"""
        return self.response

class MockAioFiles:
    def __init__(self):
        self.mock_file = AsyncMock()
        self.mock_file.write = AsyncMock()
        
    async def __aenter__(self):
        return self.mock_file
        
    async def __aexit__(self, *args):
        pass

class MockHashlib:
    def __init__(self, hex_value='test_hash'):
        self.hex_value = hex_value
        
    def hexdigest(self):
        return self.hex_value[:8]

class TestFluxGenerator(unittest.TestCase):
    @patch.dict(os.environ, {'FAL_KEY': 'test_key'}, clear=True)
    def test_generate_images_pro(self):
        """Test Pro model image generation with mocked API calls."""
        # Create mock response for successful image generation
        mock_result = {
            'images': [{
                'url': 'https://example.com/test.jpg',
                'width': 512,
                'height': 512,
                'content_type': 'image/jpeg'
            }],
            'seed': 12345,
            'has_nsfw_concepts': [False],
            'prompt': 'test prompt'
        }
        
        async def run_test():
            mock_aiofiles = MockAioFiles()
            mock_hash = MockHashlib('test_hash_value')
            
            with patch('flux_generator.fal_client') as mock_fal, \
                 patch('aiohttp.ClientSession', return_value=MockClientSession()), \
                 patch('os.makedirs') as mock_makedirs, \
                 patch('os.path.exists', return_value=False), \
                 patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
                 patch('os.getcwd', return_value='/test/workspace'), \
                 patch('hashlib.sha256', return_value=mock_hash), \
                 patch('aiofiles.open', return_value=mock_aiofiles):
                
                # Set up mock for fal_client
                mock_fal.InProgress = MockInProgress
                mock_fal.subscribe_async = AsyncMock(return_value=mock_result)
                
                # Test parameters
                test_prompt = "test prompt"
                base_dir = '/test/workspace/generated'
                output_dir = f'{base_dir}/test_hash'
                
                # Test with minimal required parameters
                results = await generate_images(
                    prompt=test_prompt,
                    count=1,
                    out_dir=output_dir,
                    model="pro"
                )
                
                # Verify minimal call
                self.assertEqual(len(results), 1)
                call_args = mock_fal.subscribe_async.call_args[1]
                self.assertEqual(call_args['arguments']['prompt'], test_prompt)
                self.assertNotIn('enable_safety_checker', call_args['arguments'])
                self.assertNotIn('sync_mode', call_args['arguments'])
                
                # Reset mock
                mock_fal.subscribe_async.reset_mock()
                
                # Test with optional parameters
                results = await generate_images(
                    prompt=test_prompt,
                    count=1,
                    out_dir=output_dir,
                    model="pro",
                    w=512,
                    h=512,
                    safety_check=True,
                    sync=True
                )
                
                # Verify results with optional parameters
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]['prompt'], test_prompt)
                
                # Verify API calls with optional parameters
                call_args = mock_fal.subscribe_async.call_args[1]
                self.assertEqual(call_args['arguments']['prompt'], test_prompt)
                self.assertEqual(call_args['arguments']['image_size']['width'], 512)
                self.assertEqual(call_args['arguments']['image_size']['height'], 512)
                self.assertEqual(call_args['arguments']['enable_safety_checker'], True)
                self.assertEqual(call_args['arguments']['sync_mode'], True)
                
                # Verify file operations
                mock_aiofiles.mock_file.write.assert_called()
        
        # Run the async test
        asyncio.run(run_test())

    def test_check_api_key(self):
        """Test API key validation."""
        # Test with valid key
        with patch.dict(os.environ, {'FAL_KEY': 'test_key'}, clear=True):
            self.assertIsNotNone(check_api_key())
        
        # Test with missing key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                check_api_key()

if __name__ == '__main__':
    unittest.main() 