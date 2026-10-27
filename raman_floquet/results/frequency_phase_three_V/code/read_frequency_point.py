"""Read and losslessly reconstruct every eigenstate of a saved scan point."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frequency_solver import reconstruct_modes
from model import localization_observables


def read_point(path):
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    if bool(data["is_frozen"]):
        modes = data["frozen_modes"]
    else:
        modes = reconstruct_modes(data["sector_vectors"], data["reconstruction_order"], L=89)
    obs = localization_observables(modes)
    error = float(np.max(abs(obs["ipr_component"]-data["IPR"])))
    if error > 1e-12:
        raise RuntimeError(f"Raw state reconstruction failed IPR check: {error}")
    data["modes"] = modes
    data["mode_probabilities"] = abs(modes)**2
    data["reconstruction_IPR_max_error"] = error
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point", type=Path, help="points/V0_p00_w01.npz, for example")
    parser.add_argument("--output", type=Path, help="Optional full178 eigenvector export NPZ")
    args = parser.parse_args()
    data = read_point(args.point)
    print(f"V={float(data['V']):g}, phi={float(data['phi']):.12g}, "
          f"omega={float(data['omega']):.12g}, frozen={bool(data['is_frozen'])}")
    print(f"Modes: {data['modes'].shape}; IPR reconstruction error "
          f"{data['reconstruction_IPR_max_error']:.3g}")
    if args.output:
        np.savez_compressed(args.output, **data)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
