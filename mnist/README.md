# MNIST experiments

## Dependencies
- Python 3.6 or later
- [PyTorch](https://pytorch.org/) 0.4 or later

## Data loading
Use `--dataroot <your-local-directory-with-MNIST>` to specify the MNIST dataset directory. 
If not found, the script will download the MNIST dataset in the given directory. 
Default directory is `.datasets/`.

## Options of the code
Run: `python main.py --help` to enumerate the available options.

## Reproducing
Please refer to `exe.sh` as an example.

## Output
The outputs are stored in a (sub)directory that can be specified with the `--outdir` option.
Default output directory is `./results/test/`.

```
<args.outdir>/
├── [generator/]   # created if selected to store the generator networks (option generator_freq > 0)
├── [backup/]      # created if selected to backup all models (option backup_freq > 0)
├── fake_samples/  # created if selected to store synthetic samples (option sample_freq > 0) as `*.png` images.
├── metrics.json   # logs the metrics in JSON (JavaScript Object Notation) format
├── metrics.log
└── log.log 
```

By default the metrics are calculated every 10-th iteration, 
and logged in `metrics.json` and `metrics.log`, which files are stored in the given `args.outdir`.
