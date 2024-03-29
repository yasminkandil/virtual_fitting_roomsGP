


* [mcn](./mcn): Main program source code
* [data](./data): **Polyvore-T** datasets based on [Polyvore](https://github.com/xthan/polyvore-dataset).
* [baselines](./baselines): Compared baselines in our experiment
* [exp](./exp): Experiment details, scripts and results etc.

## Requirements


## Usage

1. Download the original [Polyvore](https://github.com/xthan/polyvore/) dataset, then unzip the file and put the `image` directory into `data` folders (or you can create a soft link for it).

2. Train

   ```sh
   cd mcn
   python train.py
   ```

3. Evaluate

   ```
   python evaluate.py
   ```

4. Visualize outfit diagnosis

   ```
   cd exp
   python diagnosis.py
   ```


5. Automatically revise outfit

   ```
   python revision.py
   ```

## Prediction Performance

Pretrained model weights can be found in the links. The train, validation and test split is provided in [data](./data/).

|                                                              |    AUC    |   FITB    |
| :----------------------------------------------------------- | :-------: | :-------: |
| Pooling                                                      |   88.35   |   57.28   |
| Concatenation                                                |   83.40   |   52.91   |
| Self-attention                                               |   79.65   |   48.60   |
| [BiLSTM](https://drive.google.com/open?id=1WaUP0X-ytZ05HYzeHmdBSzT9gcjF1c46) |   74.82   |   46.02   |
| [CSN](https://drive.google.com/open?id=1EYwtJBRMFxRDzQs7JNYQhp2TpRF2fw9r) |   84.90   |   57.06   |
| [Ours](https://drive.google.com/open?id=1WAErKHDmDfamZQt90wAOC5Db04euIeIP) | **91.90** | **64.35** |


