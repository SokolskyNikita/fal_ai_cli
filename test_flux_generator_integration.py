import unittest
import os
import subprocess
import tempfile
import shutil
import glob

class TestFluxGeneratorIntegration(unittest.TestCase):
    """Integration tests for flux_generator.py using actual CLI."""
    
    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp(prefix='flux_test_')
    
    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def run_command(self, output_dir: str, command: str) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        return subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ
        )
    
    def test_generate_single_image_pro(self):
        """Test generating a single image with Pro model."""
        output_dir = os.path.join(self.test_dir, "pro_test")
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.run_command(output_dir, f'python flux_generator.py -p "a beautiful sunset over mountains" --model pro --image-size landscape_4_3 -n 1 -od "{output_dir}"')
        
        print(f"Command output (stdout): {result.stdout}")
        print(f"Command output (stderr): {result.stderr}")
        
        self.assertEqual(result.returncode, 0, "Command failed")
        
        # Check if files were created
        files = glob.glob(os.path.join(output_dir, "*"))
        self.assertGreater(len(files), 0, "No files were created")
        
        # Check for metadata.json
        metadata_file = os.path.join(output_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_file), "metadata.json not found")
        
        # Check if at least one image was created
        images = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.assertGreater(len(images), 0, "No images were created")
    
    def test_generate_multiple_images_ultra(self):
        """Test generating multiple images with Ultra model."""
        output_dir = os.path.join(self.test_dir, "ultra_test")
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.run_command(output_dir, f'python flux_generator.py -p "cyberpunk city at night" --model ultra -a 16:9 -n 2 -od "{output_dir}"')
        
        print(f"Command output (stdout): {result.stdout}")
        print(f"Command output (stderr): {result.stderr}")
        
        self.assertEqual(result.returncode, 0, "Command failed")
        
        # Check if files were created
        files = glob.glob(os.path.join(output_dir, "*"))
        self.assertGreater(len(files), 0, "No files were created")
        
        # Check for metadata.json
        metadata_file = os.path.join(output_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_file), "metadata.json not found")
        
        # Check if at least two images were created
        images = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.assertGreaterEqual(len(images), 2, "Not enough images were created")

if __name__ == '__main__':
    unittest.main() 