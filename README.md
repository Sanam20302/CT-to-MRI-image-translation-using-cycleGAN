# CT-to-MRI-image-translation-using-cycleGAN

An ongoing project with CycleGAN-based model for CT to MRI translation, preserving anatomical
structures with cycle-consistency. Evaluated using SSIM, PSNR and enhanced fine-grained detail through
attention and hyperparameter tuning. (PyTorch, Deep Learning)

Key files:
- CT-MRI.py : For training from epoch 0.
- CT-MRI_resume.py : To resume training from where we left before.
- training_metrics.csv : Metrics throughout and after completing 500 epochs.
