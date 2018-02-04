#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='fklearn',
    version='0.0.1',
    description=(
        'The basic tool of my personal.'
    ),
    long_description=open('README.rst').read(),
    author='frankchen',
    author_email='947118251@qq.com',
    maintainer='frankchen',
    maintainer_email='947118251@qq.com',
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://frankchen.xyz',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)
