from setuptools import setup

setup(
    name='average',
    version='1.0',
    description='Moving averaging schemes (exponentially weighted, polynomial-decay).',
    long_description=open('README.rst').read(),
    url='https://github.com/crowsonkb/average',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    license='MIT',
    packages=['average'],
    install_requires=['numpy>=1.14.3'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
