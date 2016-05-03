from setuptools import setup, find_packages
import codecs


with codecs.open('README.md', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="agentnet",
    version='0.0.7',
    description="AgentNet - a library for MDP agent design and research",
    long_description=long_description,

    # Author details
    author_email="justheuristic@gmail.com",
    url="https://github.com/yandexdataschool/AgentNet",
    
    # Choose your license
    license='MIT',
    packages=find_packages(),

    classifiers=[
        # Indicate who your project is intended for
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        #'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='machine learning, reinforcement learning, MDP, POMDP, deep learning, Q-network, Markov Decision Process, experiment platform',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'theano >= 0.8.0.dev0.dev-RELEASE',
        'lasagne >= 0.2.dev1',
        'six',
    ],
)
