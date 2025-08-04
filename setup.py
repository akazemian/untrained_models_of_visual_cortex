from setuptools import setup, find_packages
import time

start_time = time.perf_counter()

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')

    return requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='untrained_models_of_visual_cortex',  # Replace with your own package name
    version='0.1.0',
    author='Atlas Kazemian',
    author_email='atlaskazemian@gmail.com',
    description='modeling primate neural responses using high dimensional untrained CNNs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/akazemian/untrained_models_of_visual_cortex',
    packages=find_packages(),
    install_requires=[
            'pillow'==10.3.0,
            'opencv-python'==4.10.0.84,
            'loguru'==0.7.2,
            'matplotlib'==3.9.0,
            'numpy'==2.0.0,
            'pandas'==2.2.2,
            'scipy'==1.13.1,
            'seaborn'==0.13.2,
            'scikit-learn'==1.5.0,
            'timm'==1.0.7,
            'torch'==2.3.1,
            'torchmetrics'==1.4.0.post0,
            'torchvision'==0.18.1,
            'tqdm'==4.66.4,
            'xarray'==2024.6.0,
            'netCDF4'==1.7.1,
            'cupy-cuda12x'==13.2.0,
            'python-dotenv'==1.0.1,
     ], # This line includes the requirements from the requirements.txt file
    classifiers=[
        "Programming Language :: Python :: 3.10.14",
        "Operating System :: Linux",
    ],
    python_requires='>=3.10.14',
)

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.3f} seconds") 