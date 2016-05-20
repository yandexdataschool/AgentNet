from setuptools import setup, find_packages
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

# loading README (TODO convert to rst?)
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

# loading version from setup.py
with codecs.open(os.path.join(here, 'agentnet/__init__.py'), encoding='utf-8') as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)
    assert version_string.startswith('0.')


setup(
    name="agentnet",
    version=version_string,
    description="AgentNet - a library for Deep Reinforcement Learning and custom recurrent network design and research",
    long_description=long_description,

    # Author details
    author_email="justheuristic@gmail.com",
    url="https://github.com/yandexdataschool/AgentNet",

    # Choose your license
    license='MIT',
    packages=find_packages(),

    classifiers=[
        # Indicate who your project is intended for
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='machine learning, reinforcement learning, MDP, POMDP, ' + \
             'deep learning, Q-network, Markov Decision Process, experiment platform',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'six',
        'lasagne',
        'theano >= 0.8.2',
        'pandas>=0.14',
        'numpy>=1.9',
        'matplotlib >= 1.4'
    ],
)
