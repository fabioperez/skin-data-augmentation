declare -a nets=("inceptionv4" "resnet152" "densenet161")
declare -a limits=(125 250 500 1000 1500)

for i in $(seq 6); do
    for net in "${nets[@]}"; do
        for lim in "${limits[@]}"; do
            python3 train.py with \
                train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
                val_root=$VAL_ROOT val_csv=$VAL_CSV \
                test_root=$TEST_ROOT test_csv=$TEST_CSV \
                model_name="$net" \
                epochs=240 \
                'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True}' \
                limit_data="$lim" \
                --name "$net"-limit-"$lim"-full-basic

            python3 train.py with \
                train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
                val_root=$VAL_ROOT val_csv=$VAL_CSV \
                test_root=$TEST_ROOT test_csv=$TEST_CSV \
                model_name="$net" \
                epochs=240 \
                limit_data="$lim" \
                --name "$net"-limit-"$lim"-no-aug
        done
    done
done
