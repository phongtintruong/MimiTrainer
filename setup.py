from pathlib import Path
from setuptools import setup

NAME = 'MimiTrainer'  # Changed package name
DESCRIPTION = 'Unified speech tokenizer for speech language model'  # You can update this if needed
URL = 'https://github.com/phongtintruong/MimiTrainer' # Update to your repository
EMAIL = 'phongtintruong@gmail.com' # Update to your email
AUTHOR = 'Trung Thanh Nguyen' # Update to your name
REQUIRES_PYTHON = '>=3.8.0'

# This part assumes your version is in MimiTrainer/__init__.py now
for line in open('mimitrainer/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # Update package names here:
    packages=['mimitrainer', 'mimitrainer.quantization', 'mimitrainer.modules', 'mimitrainer.trainer'],
    install_requires=['numpy', 'torch', 'torchaudio', 'einops', 'scipy', 'huggingface-hub', 'soundfile', 'matplotlib', 'lion_pytorch', 'accelerate', 'transformers'],
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
    ],
)