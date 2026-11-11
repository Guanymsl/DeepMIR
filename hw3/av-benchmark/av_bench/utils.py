import torch


def clean_sample_name(sample_name: str) -> str:
    # implement your own cleaning function here to map the sample name to the sample name
    if len(sample_name) == len('000000014_zxpo56cpUBU_000007-0'):
        # extract the zxpo56cpUBU portion
        example = '000000014_zxpo56cpUBU_000007-0'
        example_target = 'zxpo56cpUBU_000007'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]
    elif len(sample_name) == len('zxpo56cpUBU_000007-0'):
        # extract the zxpo56cpUBU portion
        example = 'zxpo56cpUBU_000007-0'
        example_target = 'zxpo56cpUBU_000007'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]

    elif len(sample_name) == len('Y---g-f_I2yQ_000001_0'):
        # extract the Y---g-f_I2yQ portion
        example = 'Y---g-f_I2yQ_000001_0'
        example_target = '---g-f_I2yQ_000001'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]

    elif len(sample_name) == len('zxpo56cpUBU_000007'):
        vid_name = sample_name

    else:
        return sample_name

    return vid_name


def unroll_paired_dict(gt_dict: dict,
                       pred_dict: dict,
                       cat: bool = False) -> tuple[torch.Tensor, torch.Tensor, list]:
    pred_keys_to_sample = {k: clean_sample_name(k) for k in pred_dict.keys()}
    unpaired_samples = set(gt_dict.keys()) ^ set(pred_keys_to_sample.values())

    gt_out_list = []
    pred_out_list = []
    for key, sample_name in pred_keys_to_sample.items():
        if sample_name in unpaired_samples:
            print(f'Sample {sample_name} not found in ground truth.')
            continue
        gt_out_list.append(gt_dict[sample_name])
        pred_out_list.append(pred_dict[key])

    if cat:
        return torch.cat(gt_out_list, dim=0), torch.cat(pred_out_list,
                                                        dim=0), list(unpaired_samples)
    else:
        return torch.stack(gt_out_list, dim=0), torch.stack(pred_out_list,
                                                            dim=0), list(unpaired_samples)


def unroll_dict(d: dict, cat: bool = False) -> torch.Tensor:
    out_list = []
    for k, v in d.items():
        out_list.append(v)

    if cat:
        return torch.cat(out_list, dim=0)
    else:
        return torch.stack(out_list, dim=0)
