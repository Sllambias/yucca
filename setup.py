from setuptools import setup, find_namespace_packages

setup(name='yucca',
      packages=find_namespace_packages(include=["yucca", "yucca.*"]),
      version='0.0.1',
      description='Yucca. Framework for out-of-the box Machine Learning',
      url='https://github.com/Sllambias/Yucca',
      author='Department of Computer Science, University of Copenhagen',
      author_email='llambias@live.com',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "nibabel",
            "batchgenerators>=0.23",
            "matplotlib",
            "torchmetrics",
            "wandb",
            "monai",
            "einops"
      ],
      entry_points={
          'console_scripts': [
              'yucca_preprocess = yucca.run.run_preprocessing:main',
              'yucca_train = yucca.run.run_training:main',
              'yucca_inference = yucca.run.run_inference:main',
              'yucca_evaluation = yucca.run.run_evaluation:main',
              'yucca_finetune = yucca.run.run_finetune:main',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation']
      )
