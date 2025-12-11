from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cat_processor",
    version="1.0.0",
    author="Иванова Дарья и Хромов Алексей 6401-010302D",
    author_email="dariashikotan@gmail.com",
    description="Асинхронная обработка изображений животных",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zaglushka/cat-image-processor",
    packages=find_packages(),
    package_dir={'': '.'},

    py_modules=["main", "run_tests"],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "aiohttp>=3.8.0",
        "aiofiles>=23.1.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "python-dotenv>=1.0.0",
        "pytest"
    ],
    entry_points={
        "console_scripts": [
            "cat_processor=main:main",
            "cat_processor_tests=run_tests:run_all_tests"
        ],
    },
)