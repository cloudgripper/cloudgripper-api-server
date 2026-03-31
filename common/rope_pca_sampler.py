from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def load_rope_poses(pose_file: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not pose_file.exists():
        print(f"Pose file not found: {pose_file}")
        return None

    poses = np.load(pose_file, allow_pickle=True)
    print(f"Loading: {len(poses)} samples from {pose_file}")

    valid_poses = []
    for i, pose in enumerate(poses):
        pose_arr = np.asarray(pose, dtype=float)
        if pose_arr.shape == (20, 2):
            valid_poses.append(pose_arr.flatten())
        else:
            print(f"  Skipping sample {i} with shape {pose_arr.shape}")

    if len(valid_poses) == 0:
        print("No valid poses found")
        return None

    X = np.array(valid_poses)
    print(f"Valid samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

    return X, poses


def generate_rope_pca_dataset(robot_name, dataset_dir, n_components=10):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load real data from dataset_dir
    source_pose_file = dataset_dir / f"{robot_name}_rope_poses.npy"

    if not source_pose_file.exists():
        raise FileNotFoundError(
            f"Rope pose file not found: {source_pose_file}. "
            f"Please provide the dataset at {source_pose_file}"
        )

    result = load_rope_poses(source_pose_file)
    if result is None:
        raise ValueError(f"No valid poses found in {source_pose_file}")

    X_flat, rope_poses = result
    print(f"Using real data from {source_pose_file}")

    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Compute stats
    stats = {
        "n_samples": X_flat.shape[0],
        "n_features": X_flat.shape[1],
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "components_for_95_var": int(
            np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        ),
    }

    # Save PCA results (poses are kept in source location)
    pca_file = dataset_dir / f"{robot_name}_rope_pca.npy"
    model_file = dataset_dir / f"{robot_name}_rope_pca_model.pkl"

    np.save(pca_file, X_pca)

    # Save model and metadata (same format as rope_pca_analysis.py)
    model_data = {
        "robot_name": robot_name,
        "pca": pca,
        "scaler": scaler,
        "stats": stats,
    }
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Generated PCA dataset: {pca_file}")
    print(f"Generated model file: {model_file}")
    print(f"Explained variance: {stats['cumulative_variance'][-1]:.4f}")
    print(f"95% variance at PC{stats['components_for_95_var']}")

    return X_pca, rope_poses, stats


class RopePCASampler:
    def __init__(self, robot_name, dataset_dir=None):
        # Poses are always in the source datasets folder
        poses_dir = Path(dataset_dir) if dataset_dir else DATA_DIR
        poses_file = poses_dir / f"{robot_name}_rope_poses.npy"

        # PCA files go in the same directory as poses
        pca_file = poses_dir / f"{robot_name}_rope_pca.npy"

        # Generate PCA if it doesn't exist or if loading fails
        need_generation = not pca_file.exists()

        if not need_generation:
            # Try loading - if it fails (e.g., version incompatibility), regenerate
            try:
                _ = np.load(pca_file)
                _ = np.load(poses_file, allow_pickle=True)
            except Exception as e:
                print(f"Failed to load existing PCA dataset: {e}")
                print(f"Regenerating PCA dataset for {robot_name}...")
                need_generation = True

        if need_generation:
            print(f"Generating PCA dataset for {robot_name}...")
            generate_rope_pca_dataset(robot_name, poses_dir)

        # Load the PCA data
        if not pca_file.exists():
            raise FileNotFoundError(f"PCA file not found: {pca_file}")
        self.X_pca = np.load(pca_file)

        # Load poses
        if not poses_file.exists():
            raise FileNotFoundError(f"Rope poses not found: {poses_file}")
        self._rope_poses = np.load(poses_file, allow_pickle=True)

        self._valid_indices = np.where(self.X_pca[:, 1] > 0)[0]

    def sample_y_gt_zero(self):
        idx = np.random.choice(self._valid_indices)
        pose = np.asarray(self._rope_poses[idx], dtype=float)
        return pose, int(idx)

if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent.parent / "processed_data"
    pca_files = list(dataset_dir.glob("*_rope_pca.npy"))
    if not pca_files:
        print("No PCA data found. Run rope_pca_analysis.py first.")

    robot_name = pca_files[0].name.replace("_rope_pca.npy", "")
    print(f"Using {robot_name}")

    sampler = RopePCASampler(robot_name)
    print(f"Total samples: {len(sampler.X_pca)}")
    print(f"Samples with PC2 > 0: {len(sampler._valid_indices)}")

    print("\nSampling 3 poses:")
    for i in range(3):
        pose, idx = sampler.sample_y_gt_zero()
        print(f"  Pose: {pose}")
        pc2 = sampler.X_pca[idx, 1]
        print(f"  Sample {i+1}: index={idx}, PC2={pc2:.3f}, pose shape={pose.shape}")
