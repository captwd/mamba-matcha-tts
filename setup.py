#!/usr/bin/env python
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]


class TolerantBuildExt(build_ext):
    """宽容的扩展构建：机器缺少 MSVC 编译器时跳过 Cython 扩展而不中断安装。

    跳过后运行时会自动使用 matcha/utils/monotonic_align/__init__.py 中
    的纯 Python 回退实现，仅对齐计算稍慢，合成/训练结果完全一致。
    """

    def run(self):
        try:
            super().run()
        except Exception as e:  # pylint: disable=broad-except
            self._warn(e)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:  # pylint: disable=broad-except
            self._warn(e)

    @staticmethod
    def _warn(error):
        print("=" * 70)
        print("WARNING: matcha.utils.monotonic_align.core 编译失败（通常因缺少 Microsoft C++ Build Tools）")
        print(f"         原始错误: {error}")
        print("         已跳过。运行时将自动使用纯 Python 回退实现，不影响使用。")
        print("=" * 70)


with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "matcha", "VERSION"), encoding="utf-8") as fin:
    version = fin.read().strip()


def get_requires():
    requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements, encoding="utf-8") as reqfile:
        return [str(r).strip() for r in reqfile]


setup(
    name="matcha-tts",
    version=version,
    description="🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Matcha-TTS",
    install_requires=get_requires(),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "matcha-data-stats=matcha.utils.generate_data_statistics:main",
            "matcha-tts=matcha.cli:cli",
            "matcha-tts-app=matcha.app:main",
            "matcha-tts-get-durations=matcha.utils.get_durations_from_trained_model:main",
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
    cmdclass={"build_ext": TolerantBuildExt},
    python_requires=">=3.9.0",
)
