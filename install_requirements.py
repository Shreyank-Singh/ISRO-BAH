import subprocess
import sys


def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False
    return True


def main():
    """Install all required packages"""
    packages = [
        "rasterio",
        "numpy",
        "scipy",
        "pillow",
        "pathlib",
        "matplotlib"
    ]

    print("Installing required packages for TIF processing...")

    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print(f"\nFailed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually.")
    else:
        print("\n✓ All packages installed successfully!")


if __name__ == "__main__":
    main()
