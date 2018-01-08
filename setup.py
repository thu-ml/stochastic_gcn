from setuptools import setup
from setuptools import find_packages

setup(name='gcn',
      version='1.0',
      description='Stochastic Training of Graph Convolutional Networks in Tensorflow',
      author='Jianfei Chen',
      author_email='chrisjianfeichen@gmail.com',
      url='http://ml.cs.tsinghua.edu.cn/~jianfei',
      download_url='https://github.com/thu-ml/stochastic_gcn',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow-gpu',
                        'networkx==1.11',
                        'scipy'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())
