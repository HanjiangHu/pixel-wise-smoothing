CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new --partial 0.01 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.1_7000_new --partial 0.1 --save_k_samples 7000
#CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.05_7000 --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.005_7000_new --partial 0.005 --save_k_samples 7000
#CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.05_7000 --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.005_7000_new --partial 0.005 --save_k_samples 7000
 
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_0.25_7000_new --partial 0.004363 --save_k_samples 7000
#CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_2.5_7000 --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.25_7000_new --partial 0.004363 --save_k_samples 7000
#CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_2.5_7000 --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.7_7000_new --partial 0.0122173 --save_k_samples 7000
#CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_7_7000 --partial 0.122173 --save_k_samples 7000

CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000 --partial 0.01 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000 --partial 0.01 --save_k_samples 7000

CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000 --partial 0.01 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000 --partial 0.01 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_6000 --partial 0.01 --save_k_samples 6000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_5000 --partial 0.01 --save_k_samples 5000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_4000 --partial 0.01 --save_k_samples 4000

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_exact --partial 0.01 --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_L --partial 0.01
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_L_one --partial 0.01 --not_entire --start 2

CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_exact_1000 --partial 0.01 --small_img --exact
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_entire_1000 --partial 0.01 --small_img
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.01_7000_new_sml_L_not_entire_1000 --partial 0.01 --small_img --not_entire --resol 1000

IBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.02_7000_new --partial 0.02 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias_save/metaroom/tz/tv_noise_0.05_7000_new --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_2.5_7000_new --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.5_7000_new --partial 0.008726 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_1.25_7000_new --partial 0.021815 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.25_7000_new --partial 0.004363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.05_7000 --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias_save/metaroom/ty/tv_noise_0.05_7000_new --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.05_7000 --partial 0.05 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias_save/metaroom/tx/tv_noise_0.05_7000_new --partial 0.05 --save_k_samples 7000

CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_2.5_7000_new --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias_save/metaroom/rx/tv_noise_2.5_7000 --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_0.25_7000_new --partial 0.004363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias_save/metaroom/ry/tv_noise_2.5_7000 --partial 0.04363 --save_k_samples 7000
CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_7_7000_new --partial 0.122173 --save_k_samples 7000