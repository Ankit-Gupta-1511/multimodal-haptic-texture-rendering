# How to test the dataset

To reduce the storage size of the dataset, the repository contains a compressed version of the dataset.

The repository contains 4 compressed folders:
- `datah5.zip` contains the compressed version of the dataset.
- `img.zip` contains the pictures of the textured surfaces.
- `script.zip` contains the testing notebook and associated libraries.
- `figs.zip` contains the figures of each source for each trial.

To test the dataset, follow these steps:
extract the contents each of the compressed folders in the root directory of the repository.
you should have the following directory structure:
```bash
.
├── datah5
│   ├── subject_0
│   │   ├── 7t
│   │   │   ├── subject_0_7t_std_0.h5
│   │   │   ├── subject_0_7t_std_1.h5
│   │   │   ├── ...
│   │   ├── 10t
│   │   │   ├── ...
│   │   ├── ...
│   |   ├── 120t
│   |   │   ├── ...
│   ├── subject_1
│   │   ├── ...
│   ├── ...
│   ├── subject_9
│   │   ├── ...
├── img
|   ├── 7t
|   |   ├── 7t_l_0.jpg
|   |   ├── 7t_r_0.jpg
|   |   ├── ...
|   ├── ...
├── script
|   ├── HAVdb
|   |   ├── ...
|   ├── notebook.ipynb
├── figs
|   ├── ft_sensor
|   |   ├── ...
|   ├── kistler
|   |   ├── ...
|   ├── posistions
|   |   ├── ...
```

Open the `notebook.ipynb` file in the `script` folder and run the cells to test the dataset.

