from setuptools import setup

setup(name='febo',
      version='0.0.1',
      description='Experiment Framework for Bayesian Optimization and Bandits',
      url='https://gitlab.inf.ethz.ch/kirschnj/febo',
      author='Johannes Kirschner',
      author_email='jkirschner@inf.ethz.ch',
      license='',
      packages=['febo'],
      entry_points={
          'console_scripts': [
              'febo = febo.main:main'
          ]
      },
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'coloredlogs',
          'ruamel.yaml',
          # ~ 'GPy',
          'PyYAML',
          'h5py',
          # ~ 'celery'
         # 'ConfigSpace', # required for hpolib
         # 'hpolib2==0.0.1', # hpolib
      ],
      dependency_links=[
          'https://github.com/automl/HPOlib2/tarball/master#egg=hpolib2-0.0.1',
      ])
