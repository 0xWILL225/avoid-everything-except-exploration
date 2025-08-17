from setuptools import setup, find_packages

package_name = "viz_server"

setup(
    name=package_name,
    version="0.1.0",
    package_dir={package_name: f"src/{package_name}"},
    packages=[package_name],
    py_modules=["viz_client"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "pyzmq",
        "numpy",
        "torch",
        "termcolor",
        "fasteners",
        "urdf_parser_py",
        "sensor_msgs_py",
        "pyyaml",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "viz_server = viz_server.server:main",
            "shutdown_viz_server = shutdown_viz_server:main",
        ],
    },
)
