import h5py
import os
import numpy as np
import pandas as pd
import argschema as ags


class MakeHistogramsForNmfParameters(ags.ArgSchema):
    structure = ags.fields.String()
    brain = ags.fields.String()
    normalize_option = ags.fields.String(default="max")
    output_file = ags.fields.OutputFile()


MERFISH_CPQ = {
    "609882": "/path/to/mouse_609882/cirro_folder/atlas_brain_609882_AIT21.0_mouse.cpq/",
    "609889": "/path/to/mouse_609889/cirro_folder/atlas_brain_609889_AIT21.0_mouse.cpq/",
    "638850": "/path/to/mouse_638850/cirro_folder/atlas_brain_609889_AIT21.0_mouse.cpq/",
}

BRAIN_GRIDS = {
    "609882": {
        "x_borders": np.array([-100000, -89000, -77500, -66500, -56000, -44500, -33500, -22500, -11500, -500, 10500, 21500]),
        "y_borders": np.array([50000, 60000, 68000, 75500, 84000, 92000, 99750, 108000]),
    },
    "609889": {
        "x_borders": np.array([-100000, -89000, -77500, -66500, -56000, -44500, -33500, -22500, -11500, -500, 10500, 21500]),
        "y_borders": np.array([50000, 60000, 68000, 75500, 84000, 92000, 99750, 108000]),
    },
    "638850": {
        "x_borders": np.array([-100000, -89000, -77500, -66500, -56000, -44500, -33500, -22500, -11500, -500, 10500, 21500]) + 99000,
        "y_borders": np.array([50000, 60000, 68000, 75500, 84000, 92000, 99750, 108000]) - 48000,
    }
}

BIN_STEP = {
    "str-msn": 100,
    "str-int": 100,
    "gpe": 100,
    "gpi": 100,
    "stn": 50,
    "snr": 100,
    "pf": 50,
    "cp": 100,
    "vstr_ot": 100,
}

MIN_N_PER_TTYPE = {
    "str-msn": 1000,
    "str-int": 1000,
    "gpe": 50,
    "gpi": 50,
    "stn": 40,
    "snr": 40,
    "pf": 40,
    "cp": 50,
    "vstr_ot": 50,
}

STRUCT_BY_TYPES = {
    "cp": [
        "0990 STR D1 Sema5a Gaba_1",
        "0991 STR D1 Sema5a Gaba_1",

        "0950 STR D1 Gaba_3",
        "0951 STR D1 Gaba_3",
        "0952 STR D1 Gaba_3",
        "0953 STR D1 Gaba_4",
        "0954 STR D1 Gaba_4",
        "0955 STR D1 Gaba_5",
        "0956 STR D1 Gaba_5",
        "0957 STR D1 Gaba_5",
        "0960 STR D1 Gaba_7",
        "0961 STR D1 Gaba_8",

        "0966 STR D2 Gaba_1",
        "0968 STR D2 Gaba_1",
        "0970 STR D2 Gaba_1",
        "0971 STR D2 Gaba_1",
        "0972 STR D2 Gaba_2",
        "0979 STR D2 Gaba_4",
        "0980 STR D2 Gaba_4",
        "0981 STR D2 Gaba_4",
        "0982 STR D2 Gaba_4",
        "0983 STR D2 Gaba_5",
        "0987 STR D2 Gaba_6",
        "0988 STR D2 Gaba_6",
    ]
}


STRUCT_SUBCLASSES = {
    "str-msn": [
        'OT D3 Folh1 Gaba',
        'STR D1 Gaba',
        'STR D2 Gaba',
        'STR D1 Sema5a Gaba',
        'STR-PAL Chst9 Gaba',
    ],
    "str-int": [
    ],
    "gpe": [
        'NDB-SI-MA-STRv Lhx8 Gaba',
        'GPe-SI Sox6 Cyp26b1 Gaba',
        'PAL-STR Gaba-Chol'
    ],
    "gpi": [
        'GPi Tbr1 Cngb3 Gaba-Glut',
        'ZI Pax6 Gaba',
    ],
    "stn": [
        'STN-PSTN Pitx2 Glut',
    ],
    "snr": [
        'SNr-VTA Pax5 Npas1 Gaba',
        'SNr Six3 Gaba',
        'LDT-PCG-CS Gata3 Lhx1 Gaba',
    ],
    "pf": [
        'PF Fzd5 Glut',
    ]
}

def find_tight_borders(spatial_xy, x_borders, y_borders,
        tightness_factor=0.2, min_slice_n=250):
    tight_borders = []
    select_x = spatial_xy[:, 0]
    select_y = spatial_xy[:, 1]

    counter = 0
    for xleft, xright in zip(x_borders[:-1], x_borders[1:]):
        for ybottom, ytop in zip(y_borders[:-1], y_borders[1:]):
            in_slice_mask = ((select_x >= xleft) & (select_x < xright) & (select_y >= ybottom) & (select_y < ytop))
            ncells = np.sum(in_slice_mask)
            if ncells >= min_slice_n:
                tight_borders.append((
                    np.percentile(select_x[in_slice_mask], tightness_factor),
                    np.percentile(select_x[in_slice_mask], 100 - tightness_factor),
                    np.percentile(select_y[in_slice_mask], tightness_factor),
                    np.percentile(select_y[in_slice_mask], 100 - tightness_factor),
                ))
                counter +=1
    return tight_borders


def create_histograms(spatial_xy, tight_borders, bin_step, types_in_list,
        cluster_labels):
    hist_by_type = {}

    for ttype in types_in_list:
        ttype_mask = cluster_labels.value == ttype

        hist_list = []
        for tb in tight_borders:
            x_edges = np.arange(tb[0], tb[1] + bin_step, bin_step)
            y_edges = np.arange(tb[2], tb[3] + bin_step, bin_step)

            hist = np.histogram2d(spatial_xy[ttype_mask, 0], spatial_xy[ttype_mask, 1],
                              bins=(x_edges, y_edges),
                             )
            my_hist = hist[0]
            hist_list.append(my_hist)
        hist_concat = np.hstack([h.flatten() for h in hist_list])

    #     # Save to a dictionary using the t-type names as keys
        hist_by_type[ttype] = hist_concat

    return hist_by_type

def bin_coordinates(spatial_xy, ccf_xyz, tight_borders, bin_step):
    pass


def main(args):
    struct_key = args['structure']
    brain_name = args['brain']

    print(f"Creating histograms for structure {struct_key} with brain {brain_name}")
    data_dir = MERFISH_CPQ[brain_name]
    cluster_labels = pd.read_parquet(os.path.join(data_dir, "obs/cluster_label.parquet"))
    subclass_labels = pd.read_parquet(os.path.join(data_dir, "obs/subclass_label.parquet"))

    # Find the cells in the desired subclasses
    if struct_key in STRUCT_BY_TYPES:
        types_in_list = STRUCT_BY_TYPES[struct_key]
        mask = cluster_labels.value.isin(types_in_list)
    elif struct_key in STRUCT_SUBCLASSES:
        subclass_list = STRUCT_SUBCLASSES[struct_key]
        mask = subclass_labels.value.isin(subclass_list)
        types_in_list = np.unique(cluster_labels.loc[mask, "value"])
    elif struct_key == "vstr_ot":
        subclass_list = [
            'OT D3 Folh1 Gaba',
            'STR D1 Gaba',
            'STR D2 Gaba',
            'STR D1 Sema5a Gaba',
        ]
        mask = subclass_labels.value.isin(subclass_list)
        types_in_list = np.unique(cluster_labels.loc[mask, "value"])
        types_in_list = types_in_list[~np.in1d(types_in_list, STRUCT_BY_TYPES["cp"])]
        mask = cluster_labels.value.isin(types_in_list)
    print("Number of selected t-types:", len(types_in_list))
    print("Number of selected cells in this brain:", mask.sum())

    print("Finding ROIs")
    grid_info = BRAIN_GRIDS[brain_name]
    x_borders = grid_info['x_borders']
    y_borders = grid_info['y_borders']

    spatial_xy = pd.read_parquet(os.path.join(data_dir, "obsm/spatial_cirro.parquet"))
    tight_borders = find_tight_borders(spatial_xy.loc[mask, :].values, x_borders, y_borders)

    print(f"Identified {len(tight_borders)} ROIs")

    print("Creating histograms")
    bin_step = BIN_STEP[struct_key]
    hist_by_type = create_histograms(spatial_xy.values, tight_borders, bin_step,
        types_in_list, cluster_labels)


    print("Normalizing histograms")
    ttype_counts = np.array([h.sum() for h in hist_by_type.values()])
    ttype_order = list(hist_by_type.keys())

    min_n = MIN_N_PER_TTYPE[struct_key]
    n_with_min_cells = (ttype_counts >= min_n).sum()
    print(f"{n_with_min_cells} types have at least {min_n} cells")

    hists_as_1d_arr = np.zeros((
        n_with_min_cells,
        hist_by_type[list(hist_by_type.keys())[0]].size))
    ttype_order_min_n = np.array(ttype_order)[ttype_counts >= min_n]

    normalize_option = args['normalize_option']

    for i, t in enumerate(ttype_order_min_n):
        print(t)
        if normalize_option == 'sum':
            hists_as_1d_arr[i, :] = hist_by_type[t] / hist_by_type[t].sum()
        elif normalize_option == 'max':
            hists_as_1d_arr[i, :] = hist_by_type[t] / hist_by_type[t].max()
        elif normalize_option == 'none':
            hists_as_1d_arr[i, :] = hist_by_type[t]

    print("Saving results")
    X_input = np.asfortranarray(hists_as_1d_arr.T)

    data_f = h5py.File(args['output_file'], "w")
    dset = data_f.create_dataset("data", data=X_input)
    dset.attrs['min_n'] = min_n

    tight_borders_arr = np.array(tight_borders)

    tb_dset = data_f.create_dataset("tight_borders", data=tight_borders_arr)
    tb_dset.attrs['bin_step'] = bin_step

    dt = h5py.special_dtype(vlen=str)
    ttype_names = np.array(ttype_order_min_n, dtype=dt)
    ttypes_dset = data_f.create_dataset("ttypes", data=ttype_names)

    data_f.close()


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MakeHistogramsForNmfParameters)
    main(module.args)

