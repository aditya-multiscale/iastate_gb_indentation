import numpy as np
import pandas as pd
import os
from copy import deepcopy
from glob import glob
from typing import List, Optional, Tuple
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sys


def spt_force_retrieve(
    files: list[str], force_column: str = "Ch:Load (N)"
) -> np.ndarray:
    forces = []
    for file in files:
        df = pd.read_csv(file)
        if force_column not in df.columns:
            raise ValueError(f"Column '{force_column}' not found in file '{file}'.")
        forces.append(df[force_column].values.squeeze())
    return np.stack(forces, axis=0)


def spt_load(
    files: List[str],
    max_displacement: float,
    force_column: str = "Ch:Load (N)",
    displacement_column: str = "Ch:HL1 (V)",  # Voltage is scaled displacement
    displacement_scale: float = 3.3467,  # Scale factor to convert voltage to mm
    window_length: int = 51,
    polyorder: int = 4,
    filepath: Optional[str] = None,
    threshold: float = 2.0,
    displacement_bin: float = 0.02,
    min_displacement: float = 0.0,
    suffix: str = "p02",
) -> None:
    def digitize(
        force: np.ndarray,
        displacement: np.ndarray,
        max_displacement: float,
        threshold: float = 2.0,
        displacement_bin: float = 0.02,
        min_displacement: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        bins = np.arange(
            displacement.min(), displacement.max() + displacement_bin, displacement_bin
        )
        digitized_indices = np.digitize(displacement, bins)
        force_binned = np.array(
            [
                force[digitized_indices == i].mean()
                if len(force[digitized_indices == i]) > 0
                else 0
                for i in range(1, len(bins))
            ]
        )
        force_binned_diff = np.diff(force_binned, prepend=0)
        force_binned_diff[force_binned_diff < threshold] = 0

        indx = np.argwhere(force_binned_diff).squeeze()[0]

        zero_offset = bins[indx]
        bins = bins[indx:-1] - zero_offset
        force_binned = force_binned[indx:]
        force_binned_diff = force_binned_diff[indx:]
        mask = (0 < bins) & (bins < max_displacement + displacement_bin)
        force_binned = force_binned - force_binned[0]
        force_binned = force_binned[mask]
        force_binned_diff = force_binned_diff[mask]

        bins = bins[mask]

        force = force[displacement >= zero_offset]
        displacement = displacement[displacement >= zero_offset] - zero_offset
        force = force - force[0]

        mask = (np.round(bins, 4) >= min_displacement) & (
            np.round(bins, 4) <= max_displacement
        )
        force_binned = force_binned[mask]
        force_binned_diff = force_binned_diff[mask]
        bins = bins[mask]

        return (
            np.array(bins),
            np.array(force_binned),
            np.array(force_binned_diff),
            zero_offset,
            np.array(displacement),
            np.array(force),
        )

    plt.figure(figsize=(8, 6))
    data_list = [pd.read_csv(file) for file in files]
    curated_data = {}
    for i, data in enumerate(data_list):
        displacement, force = (
            data[displacement_column].values / displacement_scale,
            data[force_column].values,
        )
        force_smooth = savgol_filter(
            force, window_length=window_length, polyorder=polyorder
        )

        (
            displacement_binned,
            force_binned,
            _,
            _,
            displacement_thresholded,
            force_thresholded,
        ) = digitize(
            force_smooth,
            displacement,
            max_displacement=max_displacement,
            threshold=threshold,
            displacement_bin=displacement_bin,
            min_displacement=min_displacement,
        )
        smoothed_data = pd.DataFrame(
            {"Displacement": displacement_thresholded, "Force": force_thresholded}
        )
        file = os.path.basename(files[i])
        file = "_".join(file.split("_")[:2]) + f"_{i + 1}_processed_{suffix}.csv"
        if filepath is not None:
            file = os.path.join(filepath, file)

        smoothed_data.to_csv(file, index=False)

        binned_data = pd.DataFrame(
            {
                "Displacement (mm)": displacement_binned,
                "Force (N)": force_binned,
            }
        )
        binned_file = file.replace(f"_processed_{suffix}.csv", f"_binned_{suffix}.csv")
        binned_data.to_csv(binned_file, index=False)

        if i == 0:
            curated_data["Displacement (mm)"] = displacement_binned
        curated_data[f"Force - {i + 1} (N)"] = force_binned

        # Plot force-dsiplacement curve

        plt.plot(
            displacement_binned,
            force_binned,
            label=os.path.basename(binned_file).replace(".csv", ""),
            alpha=0.7,
        )

    curated_data["Force (N)"] = np.mean(
        np.array(
            [curated_data[key] for key in curated_data if key.startswith("Force -")]
        ),
        axis=0,
    )
    curated_data["Force Std (N)"] = np.std(
        np.array(
            [curated_data[key] for key in curated_data if key.startswith("Force -")]
        ),
        axis=0,
    )
    curated_df = pd.DataFrame(
        {
            key: curated_data[key]
            for key in curated_data
            if not key.startswith("Force -")
        }
    )

    curated_file = (
        "_".join(os.path.basename(files[0]).split("_")[:2]) + f"_curated_{suffix}.csv"
    )
    curated_df.to_csv(
        os.path.join(filepath, curated_file),
        index=False,
    )

    plt.plot(
        curated_data["Displacement (mm)"],
        curated_data["Force (N)"],
        label="Mean Curve",
        color="black",
        linewidth=2,
    )

    plt.fill_between(
        curated_data["Displacement (mm)"],
        curated_data["Force (N)"] - curated_data["Force Std (N)"],
        curated_data["Force (N)"] + curated_data["Force Std (N)"],
        color="gray",
        alpha=0.3,
        label="Std Dev",
    )

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.title("Force-Displacement Curve: ")
    plt.legend()
    plt.grid()
    if filepath is not None:
        plt.savefig(
            filepath
            + "/"
            + "_".join(os.path.basename(files[0]).split("_")[:2]).replace(
                ".csv", f"_{suffix}.png"
            )
        )
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    CURRENT_DIR = deepcopy(os.getcwd())
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
    if len(sys.argv) > 1 and sys.argv[1] is not None:
        PROJECT_ROOT = sys.argv[1]
    BASE_DIR = f"{PROJECT_ROOT}/Datalake/Raw/"
    CURATED_DIR = f"{PROJECT_ROOT}/Datalake/Curated/"

    aeromet100_condition1_files = glob(
        os.path.join(BASE_DIR, "Aermet100_Condition1*csv")
    )
    aeromet100_condition2_files = glob(
        os.path.join(BASE_DIR, "Aermet100_Condition2*csv")
    )
    haynes230_condition1_files = glob(
        os.path.join(BASE_DIR, "Haynes230_Condition1*csv")
    )

    nrl_steel_files = glob(os.path.join(BASE_DIR, "NRL_Steel*csv"))

    spt_load(
        aeromet100_condition1_files,
        max_displacement=0.02,
        filepath=CURATED_DIR,
        threshold=0.5,
        displacement_bin=0.002,
        min_displacement=0.004,
        suffix="p02",
    )
    spt_load(
        aeromet100_condition2_files,
        max_displacement=0.02,
        filepath=CURATED_DIR,
        threshold=1.5,
        displacement_bin=0.002,
        min_displacement=0.004,
        suffix="p02",
    )
    spt_load(
        haynes230_condition1_files,
        max_displacement=0.02,
        filepath=CURATED_DIR,
        threshold=0.5,
        displacement_bin=0.002,
        min_displacement=0.004,
        suffix="p02",
    )

    nrl_steel_files_red = [
        nrl_steel_files[i] for i in range(len(nrl_steel_files)) if i != 3
    ]
    spt_load(
        nrl_steel_files_red,
        max_displacement=0.02,
        filepath=CURATED_DIR,
        threshold=1.0,
        displacement_bin=0.002,
        min_displacement=0.004,
        suffix="p02",
    )
