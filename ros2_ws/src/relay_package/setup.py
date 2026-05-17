from setuptools import find_packages, setup

package_name = 'relay_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manoj Bhatta',
    maintainer_email='bhattamanoz124@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'publish_pose = relay_package.publish_pose:main',
            'data_recorder = relay_package.data_recorder2:main',
        ],
    },
)
