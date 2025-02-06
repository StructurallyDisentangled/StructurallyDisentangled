import sklearn
import sklearn.decomposition
from PIL import Image
import time
import numpy as np
import torch
import os
import argparse
# import logging

def pca_rendering(pca_dict: dict, features_image: torch.Tensor):
    """PCA rendering function."""
    # features_images: [H, W, features_dim]
    features_image = features_image.to(pca_dict["feature_pca_mean"])
    vis_feature = (features_image.reshape(-1, features_image.shape[-1]) - pca_dict["feature_pca_mean"][None, :]) @ pca_dict["feature_pca_components"].T
    vis_feature = (vis_feature - pca_dict["feature_pca_postprocess_sub"]) / pca_dict["feature_pca_postprocess_div"]
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((features_image.shape[0], features_image.shape[1], 3)).cpu()

    return vis_feature



def calc_pca(features_dict: dict, out_dir, name="features"):
    def path_fn(x):
        return os.path.join(out_dir, x)
    k = "features"
    fit_start_time = time.time()            
    print(f'Fitting PCA for {k}')

    pca = sklearn.decomposition.PCA(3, random_state=42)
    # reshape to [H*W, 512]
    features_tensor = torch.stack(list(features_dict.values()))
    f_samples = features_tensor.reshape(-1, features_tensor.shape[-1]).to('cpu').numpy()                    
    transformed = pca.fit_transform(f_samples)
    
    print(f'PCA fit in {(time.time() - fit_start_time):0.3f}s for {k}')

    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()

    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    pca_dict = {"pca": pca, "feature_pca_mean": feature_pca_mean, "feature_pca_components": feature_pca_components,
                "feature_pca_postprocess_sub": feature_pca_postprocess_sub, "feature_pca_postprocess_div": feature_pca_postprocess_div}
    print(f'Finished PCA fit for {k}')
                
    del f_samples

    return pca_dict

    for k in features_dict.keys():
        for i in range(features_dict[k].shape[0]):
            vis_feature = pca_rendering(pca_dict, features_dict[k][i].squeeze())
            Image.fromarray((vis_feature.cpu().numpy() * 255).astype(np.uint8)).save(path_fn(f'{k}_pca_{str(i)}.png'))

    print(f'Saved PCA for {k}')



if __name__ == "__main__":
    """
    example for run:
    python ./scripts/calc_pca.py --features_path /mnt/data/features 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True, help='path to the features directory')    
    parser.add_argument('--name', type=str, default='features', help='name of the features')

    args = parser.parse_args() 

    # load all features from features_path into a tensor 
    # loop over all files in the features_path
    

    output_path = args.features_path + '/pca'

    # Check if the folder exists
    if not os.path.exists(output_path):
        # Create the folder
        os.makedirs(output_path)
        print(f"Folder path created.")
    else:
        print(f"Folder path already exists.")

    features_tensor = []
    features_dict = {}    

    # features_types = ["cam_features", "dep_features", "indep_features"]
    # features_types = ["total_features"]
    features_types = ["indep_features"]#, "indep_features", "dep_features"]#, "total_features", "indep_features"]
    # features_dict_name = {f:{} for f in features_types}

    print(f'Loading features...')
    for f_type in features_types:
        f_path = os.path.join(args.features_path, f_type)
        for i, file in enumerate(os.listdir(f_path)):
            if ".pt" not in file:
                continue
            features = torch.load(os.path.join(f_path, file), map_location=torch.device('cpu'))
            # features_dict_name[f_type][file.split('.')[0]] =  # append the name of the file
            features_tensor.append(features)
        features_dict[f_path.split('/')[-1]] = torch.stack(features_tensor)
        features_tensor = []
    
    # features_tensor = torch.stack(features_tensor)
    print(f'finished loading features')

    print(f'Calculating PCA')
    pca_dict = calc_pca(features_dict, output_path, name=args.name)
    
    print(f'Finished PCA calculation')
    def path_fn(x):
        return os.path.join(output_path, x)
    
    for f_type in features_types:
        f_path = os.path.join(args.features_path, f_type)
        for i, file in enumerate(os.listdir(f_path)):
            if ".pt" not in file:
                continue
            features = torch.load(os.path.join(f_path, file), map_location=torch.device('cpu'))
            name = file.split('.')[0]
            vis_feature = pca_rendering(pca_dict, features)
            print(f'Saving {name}...')
            Image.fromarray((vis_feature.cpu().numpy() * 255).astype(np.uint8)).save(path_fn(f'{name}_pca.png'))