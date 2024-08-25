import numpy as np # type: ignore

def load_pref_datasets(file_path):

    data = np.load(file_path)

    return data


file_path = '/root/RL/impl_research/saves_iql_1/20240824T231027_0_827407.npz'  # Đường dẫn tới tệp .npz đã lưu
data = load_pref_datasets(file_path)
print("Keys in the .npz file:")
print(data.files)
print("obs shape:", data['obs'].shape)
print("action shape:", data['action'].shape)


print("==================================================")
file_path_2 = '/root/RL/impl_research/datasets/research/hopper2.npz'
data2 = load_pref_datasets(file_path_2)
print("Keys in the .npz file:")
print(data2.files)
print("obs_1 shape:", data2['obs_1'].shape)
print("obs_2 shape:", data2['obs_2'].shape)
print("action_1 shape:", data2['action_1'].shape)
print("action_2 shape:", data2['action_2'].shape)
print("label shape:", data2['label'].shape)
print("label data:", data2['label'])
print("label data:", data2['action_1'])
