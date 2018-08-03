declare -a nets=("inceptionv4" "resnet152" "densenet161")

for i in $(seq 6); do
    for net in "${nets[@]}"; do

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"rotation": 90, "scale": (0.8, 1.2), "shear": 20}' \
            --name "$net"-affine

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3}' \
            --name "$net"-color-general

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1}' \
            --name "$net"-color-all

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"vflip": True, "hflip": True}' \
            --name "$net"-flip

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"random_crop": True}' \
            --name "$net"-random-crop

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            --name "$net"-no-aug

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"random_erasing": True}' \
            --name "$net"-random-erasing

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"tps": True}' \
            --name "$net"-tps

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True}' \
            --name "$net"-full-basic

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True, "random_erasing": True}' \
            --name "$net"-full-erasing

        python3 train.py with \
            train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
            val_root=$VAL_ROOT val_csv=$VAL_CSV \
            test_root=$TEST_ROOT test_csv=$TEST_CSV \
            model_name="$net" \
            epochs=120 \
            'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True, "tps": True}' \
            --name "$net"-full-tps

    done
done
