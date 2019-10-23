from setuptools import setup

setup(name='sbcd',
      version='0.0.1',
      description='Semiparametric IDS',
      url='https://gitlab.inf.ethz.ch/kirschnj/febo',
      author='Johannes Kirschner',
      author_email='jkirschner@inf.ethz.ch',
      license='',
      packages=['sbcd'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          # 'tensorflow',
          'febo',
      ],
)
