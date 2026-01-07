import subprocess

# Install required packages
packages = ["transformers", "accelerate", "scikit-learn", "datasets", "pandas"]
for package in packages:
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call(['pip', 'install', package, '-U', '-q'])
        print(f"{package} installed.")

print("All required packages are installed.")