# SVRE experiments

## Dependencies
- Python 3.6 or later
- [PyTorch](https://pytorch.org/) 0.4 or later

## Data loading
Use `--data_dir <local-directory-with-selected-dataset>` to specify directory where the selected dataset is downloaded. 
If not found, the script will download the selected dataset in the `data_dir` directory, except for [ImageNet](http://image-net.org/download-images) which needs to be downloaded beforehand. 
Default directory is `.data/`.

## Options of the code
Run: `python main.py --help` to enumerate the available options.

## Reproducing
Please refer to `exe.sh` as an example.

## Output
Output is stored in a (sub)directory that can be specified with the `--version` option.
Default output directory is `./results/<dataset>/<extra/gan>[_svrg]_<selected-optimizer>[_avgFreq]/G<g_lr>_D<d_lr>[_beta<beta1>][_gamma<lr_scheduler_param>]`.

```
<args.outdir>/
├── [backup/]   # created if selected to backup nets and optim states (option backup_freq > 0)
├── logs/ 
├── models/  # dir to store G nets for IS&FID metrics (frequency def. by option model_save_step) as `*.pth`.
    ├── gen/
    ├── gen_avg/
    └── gen_ema/
└── samples/  # dir to store synthetic samples (frequency def. by option sample_step) as `*.png`.
    ├── gen/
    ├── gen_avg/
    └── gen_ema/
```
