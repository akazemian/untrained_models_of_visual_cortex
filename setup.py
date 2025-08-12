from setuptools import setup, find_packages
import time

start_time = time.perf_counter()

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
            'pillow',
            'opencv-python',
            'loguru',
            'matplotlib',
            'numpy',
            'pandas',
            'scipy',
            'seaborn',
            'scikit-learn',
            'timm',
            'torch',
            'torchmetrics',
            'torchvision',
            'tqdm',
            'xarray',
            'netCDF4',
            'cupy-cuda12x',
            'umap'
     ], # This line includes the requirements from the requirements.txt file
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: Linux",
    ],
    python_requires='>=3.12',
)

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.3f} seconds") 



# def read_requirements():
#     with open('requirements.txt') as req:
#         content = req.read()
#         requirements = content.split('\n')

#     return requirements