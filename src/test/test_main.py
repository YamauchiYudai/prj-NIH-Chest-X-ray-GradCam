import unittest
import os
import csv
import subprocess
import shutil

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup dummy data for the test."""
        # Use a dedicated directory for test data
        cls.test_data_root = os.path.join("src", "test", "test_data")
        if os.path.exists(cls.test_data_root):
            shutil.rmtree(cls.test_data_root)
        os.makedirs(cls.test_data_root, exist_ok=True)
        cls.create_mini_metadata(cls.test_data_root)

    @staticmethod
    def create_mini_metadata(data_root):
        """Creates dummy list files and CSV to allow dataset loading."""
        # Create dummy filenames that don't necessarily exist (dataset.py will use dummy image)
        train_files = [f"dummy_train_{i:03d}.png" for i in range(5)]
        test_files = [f"dummy_test_{i:03d}.png" for i in range(2)]
        
        with open(os.path.join(data_root, "train_val_list.txt"), "w") as f:
            f.write("\n".join(train_files))
            
        with open(os.path.join(data_root, "test_list.txt"), "w") as f:
            f.write("\n".join(test_files))
            
        # Create dummy CSV
        csv_path = os.path.join(data_root, "Data_Entry_2017.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Finding Labels"])
            for fname in train_files + test_files:
                writer.writerow([fname, "Cardiomegaly"])
        
        print(f"Created dummy metadata in {data_root}")

    def test_main_execution(self):
        """Runs main.py as a subprocess to verify the full pipeline."""
        # We need to pass the absolute path for docker environment if needed, 
        # but relative path 'src/test/test_data' works if running from root.
        
        cmd = [
            "python", "main.py",
            "epochs=1",
            "device=cpu",
            "dataset.batch_size=2",
            "dataset.num_workers=0",
            f"dataset.data_dir={self.test_data_root}"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output for debugging in CI
        print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)
        
        self.assertEqual(result.returncode, 0, f"main.py failed with return code {result.returncode}")
        self.assertIn("Execution finished", result.stdout)
        self.assertIn("Test Binary Accuracy", result.stdout)

    def test_pickle_pipeline(self):
        """Runs create_picklefiles.py and then main.py with use_pickle=true."""
        print("\n=== Testing Pickle Pipeline ===")
        
        # 1. Run create_picklefiles
        cmd_prep = [
            "python", "-m", "src.preprocessing.create_picklefiles",
            f"dataset.data_dir={self.test_data_root}",
            "dataset.image_size=32" # Small size for speed
        ]
        print(f"Running prep: {' '.join(cmd_prep)}")
        res_prep = subprocess.run(cmd_prep, capture_output=True, text=True)
        print("Prep STDOUT:\n", res_prep.stdout)
        if res_prep.stderr: print("Prep STDERR:\n", res_prep.stderr)
        
        self.assertEqual(res_prep.returncode, 0, f"create_picklefiles failed: {res_prep.stderr}")
        
        # Verify pickle files exist
        pickle_dir = os.path.join(self.test_data_root, "pickles")
        self.assertTrue(os.path.exists(pickle_dir))
        files = os.listdir(pickle_dir)
        print(f"Created pickle files: {files}")
        self.assertTrue(len(files) > 0)
        
        # 2. Run main with use_pickle=true
        cmd_main = [
            "python", "main.py",
            "epochs=1",
            "device=cpu",
            "dataset.batch_size=2",
            "dataset.num_workers=0",
            f"dataset.data_dir={self.test_data_root}",
            "dataset.use_pickle=true",
            "dataset.image_size=32" # Must match prep
        ]
        print(f"Running main with pickle: {' '.join(cmd_main)}")
        res_main = subprocess.run(cmd_main, capture_output=True, text=True)
        print("Main STDOUT:\n", res_main.stdout)
        if res_main.stderr: print("Main STDERR:\n", res_main.stderr)
        
        self.assertEqual(res_main.returncode, 0, f"main.py with pickle failed: {res_main.stderr}")
        self.assertIn("Loading data from Pickle files", res_main.stdout)
        self.assertIn("Test Binary Accuracy", res_main.stdout)

if __name__ == '__main__':
    unittest.main()
