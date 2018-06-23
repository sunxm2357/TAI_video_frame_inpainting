#!/bin/bash

# generate plots fig4|fig7(a)|fig8|fig9|fig24
python src/util/plots.py --which_plot fig4
python src/util/plots.py --which_plot fig7\(a\)
python src/util/plots.py --which_plot fig8
python src/util/plots.py --which_plot fig9
python src/util/plots.py --which_plot fig24

# generate visualizations for KTH Actions|UCF-101|HMDB-51
python src/util/images.py --dataset KTH\ Actions
python src/util/images.py --dataset UCF-101
python src/util/images.py --dataset HMDB-51

