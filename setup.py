from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        import subprocess, os
        subprocess.check_call(["cmake", ".", "-Bbuild"])
        subprocess.check_call(["cmake", "--build", "build"])
        build_ext.run(self)

setup(
    name="excitation-rk4-sparse",
    version="0.1.0",
    packages=["python"],
    package_dir={"": "python"},
    cmdclass={"build_ext": CMakeBuild},
)
