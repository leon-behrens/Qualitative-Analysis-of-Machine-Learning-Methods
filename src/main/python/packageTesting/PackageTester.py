import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.models as models
from torch.nn import Linear
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
import csv
from datetime import datetime
import pickle
from IPython.display import clear_output
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import logging.config
import json
import tempfile


def check_packages():
    results = {}

    # Test individual imports
    import_tests = {
        "os": lambda: os.name,
        "torch": lambda: torch.__version__,
        "diffusers.StableDiffusionPipeline": lambda: StableDiffusionPipeline,
        "PIL.Image": lambda: Image,
        "torch.utils.data.Dataset": lambda: Dataset,
        "torchvision.models": lambda: models,
        "torch.nn.Linear": lambda: Linear,
        "torchvision.models.ResNet50_Weights": lambda: ResNet50_Weights,
        "torch.nn": lambda: nn,
        "torch.nn.functional": lambda: F,
        "csv": lambda: csv,
        "datetime": lambda: datetime,
        "pickle": lambda: pickle,
        "IPython.display.clear_output": lambda: clear_output,
        "random": lambda: random.randint(0, 10),
        "torchvision.transforms": lambda: transforms,
        "matplotlib.pyplot": lambda: plt,
        "pandas": lambda: pd,
        "shutil": lambda: shutil,
        "logging.config": lambda: logging.config,
        "json": lambda: json.dumps({}),
        "tempfile": lambda: tempfile.gettempdir(),
    }

    # Test each import
    for package, test in import_tests.items():
        try:
            test()
            results[package] = "Working"
        except Exception as e:
            results[package] = f"Error: {e}"

    # Specific functionality tests (from your script)
    # torch
    try:
        torch_version = torch.__version__
        results['torch (functionality)'] = f"Working (version {torch_version})"
    except Exception as e:
        results['torch (functionality)'] = str(e)

    # diffusers
    try:
        _ = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        results['diffusers (functionality)'] = "Working"
    except ImportError as e:
        results['diffusers (functionality)'] = "Requires `transformers` library. Install with `pip install transformers`."
    except Exception as e:
        results['diffusers (functionality)'] = str(e)

    # PIL
    try:
        img = Image.new('RGB', (10, 10))
        results['PIL (functionality)'] = "Working"
    except Exception as e:
        results['PIL (functionality)'] = str(e)

    # torchvision models
    try:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        results['torchvision.models (functionality)'] = "Working"
    except Exception as e:
        results['torchvision.models (functionality)'] = str(e)

    # matplotlib
    try:
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.close()
        results['matplotlib (functionality)'] = "Working"
    except Exception as e:
        results['matplotlib (functionality)'] = str(e)

    # pandas
    try:
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        results['pandas (functionality)'] = "Working"
    except Exception as e:
        results['pandas (functionality)'] = str(e)

    # json
    try:
        json_data = json.dumps({"key": "value"})
        _ = json.loads(json_data)
        results['json (functionality)'] = "Working"
    except Exception as e:
        results['json (functionality)'] = str(e)

    return results


if __name__ == "__main__":
    package_results = check_packages()
    output_file = "package_check_results.txt"

    # Write results to a text file
    with open(output_file, "w") as f:
        for package, status in package_results.items():
            f.write(f"{package}: {status}\n")

    print(f"Results written to {output_file}")
