from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dynamixel_fingers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armlab',
    maintainer_email='armlab@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'optical_flow_ros = dynamixel_fingers.optical_flow_ros:main',
            'rubbing_motion = dynamixel_fingers.rubbing_motion:main',
            'save_wrench_data = dynamixel_fingers.save_wrench_data:main',
            'depth_vid_saver = dynamixel_fingers.depth_vid_saver:main',
        ],
    },
)

    # name='dynamixel_fingers',
    # version='0.0.0',
    # packages=['dynamixel_fingers'],
    # entry_points={
    #     'console_scripts': [
    #         'rubbing_motion = dynamixel_fingers.rubbing_motion:main',
    #     ],
    # },
