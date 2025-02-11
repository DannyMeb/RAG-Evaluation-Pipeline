# Project: Rag systems evaluation pipeline using Ragas
# Authored by Daniel
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.


from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the dependencies from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ragas_project",  # Name of your package
    version="1.0.0",
    author="Daniel Meb",
    author_email="your.email@example.com",
    description="A project for generating and evaluating synthetic test sets using AI models like Falcon and Llama.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/airc-root/airc/edge_llm/falcon-rag.git",  # GitLab repository URL
    packages=find_packages('src'),  # Look for packages in the 'src' directory
    package_dir={'': 'src'},  # Specify 'src' as the root directory for packages
    include_package_data=True,
    install_requires=required,  # Automatically read from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or another license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
