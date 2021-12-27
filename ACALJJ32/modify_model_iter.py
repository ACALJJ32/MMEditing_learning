import torch

def modify_iter(model_path, iter = 0):
    state_dict = torch.load(model_path)
    print("Model Iters: ", state_dict['meta']['iter'])

    print("Modified iter from {} ==> {}.".format(state_dict['meta']['iter'], iter))
    state_dict['meta']['iter'] = iter

    torch.save(state_dict, './modified_model_{}.pth'.format(iter))


if __name__ == "__main__":
    model_path = "./edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth"
    modify_iter(model_path=model_path)