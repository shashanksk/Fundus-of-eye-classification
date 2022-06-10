mask = cv2.cvtColor(cv2.imread(item.replace(image_path, mask_path).replace(
            image_prefix, mask_prefix)), cv2.COLOR_BGR2RGB)