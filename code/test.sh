# test fully_supervised_swinunet on ACDC
python test_2D_ACDC.py --root_path ../datasets/ACDC --model swinunet --save_path ../model_best.pth
# test fully_supervised_unet on ACDC
python test_2D_ACDC.py --root_path ../datasets/ACDC --model unet --save_path ../model_best.pth
# test inherent_consistent_swinunet on ACDC
python test_2D_ACDC.py --root_path ../datasets/ACDC --model swinunet --save_path ../model_best.pth
# test inherent_consistent_unet on ACDC
python test_2D_ACDC.py --root_path ../datasets/ACDC --model unet --save_path ../model_best.pth


# test fully_supervised_swinunetr on BraTS
python test_3D_BraTS.py --root_path ../datasets/BraTS19 --model swinunetr --save_path ../model_best.pth
# test fully_supervised_unet_3D on BraTS
python test_3D_BraTS.py --root_path ../datasets/BraTS19 --model unet_3D --save_path ../model_best.pth
# test inherent_consistent_swinunet on BraTS
python test_3D_BraTS.py --root_path ../datasets/BraTS19 --model swinunetr --save_path ../model_best.pth
# test inherent_consistent_unet on BraTS
python test_3D_BraTS.py --root_path ../datasets/BraTS19 --model unet_3D --save_path ../model_best.pth

# test fully_supervised_unet_3D on AMOS
python test_3D_AMOS.py --root_path ../datasets/AMOS --save_path ../model_best.pth
# test inherent_consistent_unet_3D on AMOS
python test_3D_AMOS.py --root_path ../datasets/AMOS --save_path ../model_best.pth