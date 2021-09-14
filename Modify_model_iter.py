import torch

state_dict = torch.load('/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/VSR/mmediting_gky/work_dirs/basicvsr_reds4_tencent/modify.pth')

print('Iter: ', state_dict['meta']['iter'])

# print('modify iter to 0')

# state_dict['meta']['iter'] = 0

# print('Iter: ', state_dict['meta']['iter'])

# torch.save(state_dict, './modify.pth')