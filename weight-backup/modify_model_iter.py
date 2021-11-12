import torch

state_dict = torch.load('./edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth')

print('Iter: ', state_dict['meta']['iter'])

# print('modify iter to 0')

state_dict['meta']['iter'] = 0

# print('Iter: ', state_dict['meta']['iter'])

torch.save(state_dict, './edvrm_x4_8x4_600k_reds.pth')
