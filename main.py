from data import LearningTask


if __name__ == '__main__':
    task = LearningTask(d=20, g_vec_max_radius=0.1, x_max_radius=0.1, noise_min=-0.01, noise_max=0.01)
    n_train_samp = 50
    train_set = task.get_dataset(n_train_samp)
