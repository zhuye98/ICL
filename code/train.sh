# train fully_supervised_swinunet on ACDC
python -u train_fully_supervised_swinunet_2D_ACDC.py --root_path ../datasets/ACDC --labeled_num 3 --max_iterations 30000 --base_lr 0.1 --num_tries 1 &&
python -u train_fully_supervised_swinunet_2D_ACDC.py --root_path ../datasets/ACDC --labeled_num 7 --max_iterations 30000 --base_lr 0.1 --num_tries 1 
# train fully_supervised_unet on ACDC
python -u train_fully_supervised_unet_2D_ACDC.py --root_path ..datasets/ACDC --labeled_num 3 --max_iterations 30000 --base_lr 0.1 --num_tries 1 &&
python -u train_fully_supervised_unet_2D_ACDC.py --root_path ..datasets/ACDC --labeled_num 7 --max_iterations 30000 --base_lr 0.1 --num_tries 1

# train inherent_consistent_swinunet on ACDC
python -u train_inherent_consistent_swinunet_2D.py --root_path ../datasets/ACDC --labeled_num 3 --max_iterations 30000 --base_lr 0.1 --num_tries 1 &&
python -u train_inherent_consistent_swinunet_2D.py --root_path ../datasets/ACDC --labeled_num 7 --max_iterations 30000 --base_lr 0.1 --num_tries 1
# train inherent_consistent_unet on ACDC
python -u train_inherent_consistent_unet_2D.py --root_path ../datasets/ACDC --labeled_num 3 --max_iterations 30000 --base_lr 0.1 --num_tries 1 &&
python -u train_inherent_consistent_unet_2D.py --root_path ../datasets/ACDC --labeled_num 7 --max_iterations 30000 --base_lr 0.1 --num_tries 1

# train fully_supervised_swinunetr on BraTS
python -u train_fully_supervised_swinunetr_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 25  --max_iterations 30000 --base_lr 0.01 --num_tries 1 &&
python -u train_fully_supervised_swinunetr_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 50  --max_iterations 30000 --base_lr 0.01 --num_tries 1
# train fully_supervised_unet on BraTS
python -u train_fully_supervised_unet_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 25  --max_iterations 30000 --base_lr 0.01 --num_tries 1 &&
python -u train_fully_supervised_unet_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 50  --max_iterations 30000 --base_lr 0.01 --num_tries 1

# train inherent_consistent_swinunet on BraTS
python -u train_inherent_consistent_swinunetr_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 25  --max_iterations 30000 --base_lr 0.01 --num_tries 1 &&
python -u train_inherent_consistent_swinunetr_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 50  --max_iterations 30000 --base_lr 0.01 --num_tries 1
# train inherent_consistent_unet on BraTS
python -u train_inherent_consistent_unet_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 25  --max_iterations 30000 --base_lr 0.01 --num_tries 1 &&
python -u train_inherent_consistent_unet_3D_BraTS.py --root_path ../datasets/BraTS19 --labeled_num 50  --max_iterations 30000 --base_lr 0.01 --num_tries 1

# train fully_supervised_3D on AMOS
python train_fully_supervised_unet_3D_AMOS22.py --root_path ..datasets/AMOS --split_path ..datasets/AMOS/dataset_semi_ct.json --labeled_num 15 --val_num 30 --max_iterations 60000 --base_lr 0.02 --num_tries 1 &&
python train_fully_supervised_unet_3D_AMOS22.py --root_path ..datasets/AMOS --split_path ..datasets/AMOS/dataset_semi_ct.json --labeled_num 30 --val_num 30 --max_iterations 60000 --base_lr 0.02 --num_tries 1
# train inherent_consistent_unet_3D on AMOS
python train_inherent_consistent_unet_3D_AMOS22.py --root_path ..datasets/AMOS --split_path ..datasets/AMOS/dataset_semi_ct.json --labeled_num 15 --val_num 30 --max_iterations 60000 --base_lr 0.02 --num_tries 1 &&
python train_inherent_consistent_unet_3D_AMOS22.py --root_path ..datasets/AMOS --split_path ..datasets/AMOS/dataset_semi_ct.json --labeled_num 30 --val_num 30 --max_iterations 60000 --base_lr 0.02 --num_tries 1

