from setuptools import setup, find_packages

setup(
    name="phrase-labeler",
    version="0.1.2",
    packages=find_packages(
        include=['phrase_labeler', 'phrase_labeler.*']
    ),
    install_requires=[
        "openai"       
    ],
    entry_points={
        'console_scripts': [
            'label-phrase=phrase_labeler.phrase_labeler:main',
        ],
    },
    include_package_data=True,
    description="A package to label sentence segments given predefined segment labels using OpenAI API",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Ziwei Gu",
    author_email="ziweigu@g.harvard.edu",
    url="https://github.com/ZiweiGu/GP-TSM", 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
