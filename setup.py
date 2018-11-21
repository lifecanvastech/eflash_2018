from setuptools import setup

version = "0.1.0"

setup(
    name="eflash-2018",
    version=version,
    description="Software used in the EFlash paper",
    install_requires=[
        "keras",
        "tensorflow",
        "scikit-learn",
        "scikit-image"
    ],
    author="Kwanghun Chung Lab",
    packages=["eflash_2018"],
    entry_points=dict(console_scripts=[
        "find-pv-centers=eflash_2018.find_pv_centers:main",
        "detect-blobs=eflash_2018.detect_blobs:main",
        "collect-patches=eflash_2018.collect_patches:main"
    ])
)