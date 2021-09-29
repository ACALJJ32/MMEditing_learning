import torch

state_dict = torch.load('./iter_latest.pth')

print('Iter: ', state_dict['meta']['iter'])

# print('modify iter to 0')

# state_dict['meta']['iter'] = 0

# print('Iter: ', state_dict['meta']['iter'])

# torch.save(state_dict, './modify.pth')
