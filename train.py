# import necessary modules
import argparse
import json
import utils

from data import AxialPatchDataset, AxialSliceDataset
from model import Trainer

if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The configuration file')

    args = parser.parse_args()
    config = json.load(open(args.config))
    print(json.dumps(config, sort_keys=True, indent=4))

    # initialize
    general = config["GENERAL"]
    utils.set_gpu(general["gpu"])
    utils.seed_everything(general["seed"])

    # set loaders
    data_train = [AxialPatchDataset(config["DATA"], opt="low"),
                  AxialPatchDataset(config["DATA"]),
                  AxialPatchDataset(config["DATA"], opt="high")]
    data_val = [AxialSliceDataset(config["DATA"], mode="val", opt="low"),
                AxialSliceDataset(config["DATA"], mode="val"),
                AxialSliceDataset(config["DATA"], mode="val", opt="high")]

    # train
    model = Trainer(config)
    model.fit(data_train, data_val)
