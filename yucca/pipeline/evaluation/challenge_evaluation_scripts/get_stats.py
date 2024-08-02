# %%
from batchgenerators.utilities.file_and_folder_operations import load_json
import argparse

path = "/zhome/af/0/210164/yucca_data/results/Task017_BTCV/Task017_BTCV/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_0/version_0/lastVal/results_SURFACE2.json"


def get_dice_and_nsd(path):
    file = load_json(path)
    dice_vals = []
    nsd_vals = []
    for i in file["mean"].keys():
        if i in [0, "0"]:
            continue
        dice_vals.append(file["mean"][i]["Dice"])
        nsd_vals.append(file["mean"][i]["Average Surface Distance"])

    print("\nDICE:", dice_vals)
    print(f"=AVERAGE{tuple(dice_vals)} \n")
    print("NSD: ", nsd_vals)
    print(f"=AVERAGE{tuple(nsd_vals)} \n")


# get_dice_and_nsd(path)


def main():
    parser = argparse.ArgumentParser()

    # Required Arguments #
    parser.add_argument("--path")
    args = parser.parse_args()

    get_dice_and_nsd(args.path)


if __name__ == "__main__":
    main()
# %%
