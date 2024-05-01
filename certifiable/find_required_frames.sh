CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_exact --partial 0.01  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_entire --partial 0.01 
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_not_entire --partial 0.01 --not_entire 

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.25_7000_new_sml_L_exact --partial 0.004363  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.25_7000_new_sml_L_entire --partial 0.004363
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.25_7000_new_sml_L_not_entire --partial 0.004363  --not_entire

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.005_7000_new_sml_L_exact --partial 0.005  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.005_7000_new_sml_L_entire --partial 0.005 
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.005_7000_new_sml_L_not_entire --partial 0.005  --not_entire 
 
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.005_7000_new_sml_L_exact --partial 0.005  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.005_7000_new_sml_L_entire --partial 0.005 
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.005_7000_new_sml_L_not_entire --partial 0.005  --not_entire 

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_0.25_7000_new_sml_L_exact --partial 0.004363  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_0.25_7000_new_sml_L_entire --partial 0.004363
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_0.25_7000_new_sml_L_not_entire --partial 0.004363  --not_entire

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.7_7000_new_sml_L_exact --partial 0.0122173  --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.7_7000_new_sml_L_entire --partial 0.0122173
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.7_7000_new_sml_L_not_entire --partial 0.0122173  --not_entire