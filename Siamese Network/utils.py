def train_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM,DIM),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ]
    )

def valid_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM,DIM),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )

def append_to_data(path, image, imgs, label, data):
    image_path = os.path.join(path, image)
    for img in imgs:
        img_path = os.path.join(path, img)
        data.append([image_path, img_path, label])
    return data

def create_dataset(path, s_factor, ds_factor):
    data = []   

    folders = os.listdir(path)
    if 'README' in folders:
        folders.remove('README')

    file_names = {}
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        images = [os.path.join(folder, file) for file in files if file.endswith('.png')]
        file_names[folder] = images

    for folder in tqdm(folders):
        images = file_names[folder]
        temp = folders.copy()
        temp.remove(folder)
        for image in images:
            imgs = random.sample(images, s_factor + 1)
            if image in imgs:
                imgs.remove(image)
            else:
                imgs = imgs[:-1]
            data = append_to_data(path, image, imgs, 1, data)

            sel_fs = random.sample(temp, ds_factor)
            imgs = []
            for f in sel_fs:
                imgs.append(random.sample(file_names[f], 1)[0])
            data = append_to_data(path, image, imgs, 0, data)
    
    return data

def show_metrics(trainer):
    metrics = pd.read_csv('{}/metrics.csv'.format(trainer.logger.log_dir))

    fig, axes = plt.subplots(1,2, figsize = (12,4))

    train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
    val_loss = metrics['val_loss'].dropna().reset_index(drop=True)

    axes[0].grid(True)
    axes[0].plot(train_loss, color="r", marker="o", label='train/loss')
    axes[0].plot(val_loss, color="b", marker="x", label='valid/loss')
    axes[0].legend(loc='upper right', fontsize=9)

    lr = metrics['lr'].dropna().reset_index(drop=True)

    axes[1].grid(True)
    axes[1].plot(lr, color="g", marker="o", label='learning rate')
    axes[1].legend(loc='upper right', fontsize=9)
