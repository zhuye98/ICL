train_fully_supervised_swinunet_2D --done!
    python train_fully_supervised_swinunet_2D_ACDC.py --root_path /home/yezhu/datasets/ACDC --labeled_num 3/7 --max_iterations 30000 --base_lr 0.1
    python test_2D_ACDC.py --root_path ../datasets/ACDC --model swinunet --save_path ../model_best.pth

train_fully_supervised_unet_2D --done!
    python train_fully_supervised_unet_2D_ACDC.py --root_path /home/yezhu/datasets/ACDC --labeled_num 3
    python test_2D_ACDC.py --root_path /home/yezhu/datasets/ACDC --model unet --save_path ../model_best.pth

train_fully_supervised_swinunetr_3D_BraTS --done!
    python train_fully_supervised_swinunetr_3D_BraTS.py --root_path /home/yezhu/datasets/BraTS19 --labeled_num 25 
    python train_fully_supervised_unet_3D_BraTS.py --root_path /home/yezhu/datasets/BraTS19 --labeled_num 25

train_fully_supervised_unet_3D_AMOS22 --done!
    python train_fully_supervised_unet_3D_AMOS22.py --root_path /mntnfs/med_data2/yezhu/amos_dataset/nnUNet_raw/nnUNet_raw_data/Task_AMOS --split_path /mntnfs/med_data2/yezhu/amos_dataset/nnUNet_raw/nnUNet_raw_data/Task_AMOS/dataset_semi_ct.json --labeled_num 15 --val_num 30
    python test_3D_AMOS.py --root_path /mntnfs/med_data2/yezhu/amos_dataset/nnUNet_raw/nnUNet_raw_data/Task_AMOS --save_path ../model_best.pth

train_inherent_consistent_swinunet_2D --done!
    python train_inherent_consistent_swinunet_2D.py --root_path /home/yezhu/datasets/ACDC --labeled_num 3
    python test_2D_ACDC.py --root_path /home/yezhu/datasets/ACDC --model swinunet --save_path /home/yezhu/ICL/experiments/ACDC/Inherent_Consistent_Learning_3_labeled/icl_swinunet_exp_1/model/model_best.pth

train_inherent_consistent_unet_2D --done!
    python -u train_inherent_consistent_unet_2D.py --root_path /home/yezhu/datasets/ACDC --labeled_num 3
    python test_2D_ACDC.py --root_path /home/yezhu/datasets/ACDC --model unet --save_path /home/yezhu/ICL/experiments/ACDC/Inherent_Consistent_Learning_3_labeled/icl_unet_exp_1/model/model_best.pth

train_inherent_consistent_swinunetr_3D_BraTS
    python train_inherent_consistent_swinunetr_3D_BraTS.py --root_path /home/yezhu/datasets/BraTS19 --labeled_num 25
    python test_3D_BraTS.py --model swinunetr --root_path /home/yezhu/datasets/BraTS19 --save_path /home/yezhu/ICL/experiments/BraTS19/Inherent_Consistent_Learning_25_labeled/unet_3D_icl_exp_1/model/model_best.pth

train_inherent_consistent_unet_3D_BraTS
    python train_inherent_consistent_unet_3D_BraTS.py --root_path /home/yezhu/datasets/BraTS19 --labeled_num 25
    python test_3D_BraTS.py --model unet_3D --root_path /home/yezhu/datasets/BraTS19 --save_path /home/yezhu/ICL/experiments/BraTS19/Inherent_Consistent_Learning_25_labeled/unet_3D_icl_exp_1/model/model_best.pth

train_inherent_consistent_unet_3D_AMOS22
    python train_inherent_consistent_unet_3D_AMOS22.py --root_path /mntnfs/med_data2/yezhu/amos_dataset/nnUNet_raw/nnUNet_raw_data/Task_AMOS 