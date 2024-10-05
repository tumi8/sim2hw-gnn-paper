import enum
import math
import numpy as np
import torch

PARAMS_NORMALIZATION_VALUES_OM2HVNET = {
    "flow_rate": {
        "min": 1000.0,
        "max": 831162.0,
        "mean": 102175.78169542385,
        "std": 112323.34995486778
    },
    "flow_gamma_shape": {
        "min": 0.5325625213779135,
        "max": 1.332074126203368,
        "mean": 1.096132224296692,
        "std": 0.08366515992073066
    },
    "flow_gamma_scale": {
        "min": 3.016075680917091e-06,
        "max": 0.003317930326235959,
        "mean": 0.00018517353638414607,
        "std": 0.00045421251453550443
    },
    "hvnet_link_rate": {
        "min": 413.04347826086956,
        "max": 1900.0,
        "mean": 671.8528995756718,
        "std": 195.02488880728907
    },
    "omnet_link_rate": {
        "min": 0.0,
        "max": 1000.0,
        "mean": 1000.0,
        "std": 0.0
    },
    "flow_latency_hvnet_mean": {
        "min": 31.001229748590692,
        "max": 14160.564920922061,
        "mean": 205.20020575132045,
        "std": 515.4280191835654
    },
    "flow_latency_omnet_mean": {
        "min": 3.386831424575628,
        "max": 3371.3255505160246,
        "mean": 25.478652880378238,
        "std": 242.8865446296346
    },
    "flow_latency_hvnet_min": {
        "min": 12.95,
        "max": 2105.737,
        "mean": 47.45266725146199,
        "std": 139.3754991922913
    },
    "flow_latency_omnet_min": {
        "min": 3.386,
        "max": 412.504526,
        "mean": 6.886912530994152,
        "std": 10.298895670522427
    },
    "flow_latency_hvnet_max": {
        "min": 98.013,
        "max": 1644617.362,
        "mean": 11704.092897076025,
        "std": 106094.62634583387
    },
    "flow_latency_omnet_max": {
        "min": 5.726359,
        "max": 3496.943725,
        "mean": 46.1294269511696,
        "std": 250.26675578818563
    },
    "flow_latency_hvnet_gamma_shape": {
        "min": 0.1518510618757459,
        "max": 1083.765214356513,
        "mean": 15.915377454573028,
        "std": 57.76734817199047
    },
    "flow_latency_omnet_gamma_shape": {
        "min": 1.493742764860811,
        "max": 16834.96520097552,
        "mean": 257.8485399869026,
        "std": 863.6769971469381
    },
    "flow_latency_hvnet_gamma_scale": {
        "min": 0.8606651902084574,
        "max": 51074.28755379639,
        "mean": 78.97863770931232,
        "std": 1355.9347833097886
    },
    "flow_latency_omnet_gamma_scale": {
        "min": 0.00020116695380539064,
        "max": 13.588618549520731,
        "mean": 0.47556964309572347,
        "std": 1.2611729048553246
    },
    "flow_latency_hvnet_p5": {
        "min": 24.838,
        "max": 2844.038,
        "mean": 106.5860182748538,
        "std": 250.25246141421326
    },
    "flow_latency_omnet_p5": {
        "min": 3.386,
        "max": 3220.7263602499997,
        "mean": 23.501481892660827,
        "std": 232.01038826368122
    },
    "flow_latency_hvnet_p10": {
        "min": 25.55,
        "max": 2887.825,
        "mean": 118.26004064327485,
        "std": 258.4861120352128
    },
    "flow_latency_omnet_p10": {
        "min": 3.386,
        "max": 3275.1906065000003,
        "mean": 23.799052740292403,
        "std": 235.97690609822845
    },
    "flow_latency_hvnet_p15": {
        "min": 26.4,
        "max": 2914.925,
        "mean": 128.03804567251464,
        "std": 265.77939288886773
    },
    "flow_latency_omnet_p15": {
        "min": 3.386,
        "max": 3305.5550602499998,
        "mean": 23.9768282230848,
        "std": 238.1990079625374
    },
    "flow_latency_hvnet_p20": {
        "min": 26.725,
        "max": 2935.325,
        "mean": 138.68249175438598,
        "std": 284.74054890316677
    },
    "flow_latency_omnet_p20": {
        "min": 3.386,
        "max": 3327.888423,
        "mean": 24.119089961403517,
        "std": 239.8039664668093
    },
    "flow_latency_hvnet_p25": {
        "min": 27.15,
        "max": 3067.92175,
        "mean": 146.97718092105262,
        "std": 299.5702308841741
    },
    "flow_latency_omnet_p25": {
        "min": 3.386,
        "max": 3345.02682775,
        "mean": 24.24806220255849,
        "std": 241.0524932221578
    },
    "flow_latency_hvnet_p30": {
        "min": 27.7,
        "max": 3133.6616,
        "mean": 153.7587284210526,
        "std": 304.5043693307437
    },
    "flow_latency_omnet_p30": {
        "min": 3.386,
        "max": 3359.1617185,
        "mean": 24.37156903052632,
        "std": 242.07541285932962
    },
    "flow_latency_hvnet_p35": {
        "min": 28.263,
        "max": 3171.85615,
        "mean": 160.35210377192982,
        "std": 308.9999609739882
    },
    "flow_latency_omnet_p35": {
        "min": 3.386,
        "max": 3371.07516725,
        "mean": 24.49400990671053,
        "std": 242.94084007756985
    },
    "flow_latency_hvnet_p40": {
        "min": 28.587,
        "max": 3208.042,
        "mean": 166.9567419883041,
        "std": 313.7760522927908
    },
    "flow_latency_omnet_p40": {
        "min": 3.386,
        "max": 3381.445706,
        "mean": 24.622003823157904,
        "std": 243.69004112725702
    },
    "flow_latency_hvnet_p45": {
        "min": 28.8,
        "max": 3247.1332500000003,
        "mean": 173.65838133040936,
        "std": 318.4605280977034
    },
    "flow_latency_omnet_p45": {
        "min": 3.386,
        "max": 3390.6832155,
        "mean": 24.767420782002926,
        "std": 244.35369285778225
    },
    "flow_latency_hvnet_p50": {
        "min": 29.025,
        "max": 3291.081,
        "mean": 180.68565687134503,
        "std": 323.42163417254204
    },
    "flow_latency_omnet_p50": {
        "min": 3.386,
        "max": 3398.972118,
        "mean": 24.926435414327486,
        "std": 244.94462283623074
    },
    "flow_latency_hvnet_p55": {
        "min": 29.262,
        "max": 3324.11755,
        "mean": 189.05651809941523,
        "std": 329.5299843366849
    },
    "flow_latency_omnet_p55": {
        "min": 3.386,
        "max": 3406.425665,
        "mean": 25.10681166897661,
        "std": 245.47682905667472
    },
    "flow_latency_hvnet_p60": {
        "min": 29.5,
        "max": 3352.3366,
        "mean": 198.15758175438597,
        "std": 336.78697677168725
    },
    "flow_latency_omnet_p60": {
        "min": 3.386,
        "max": 3413.230377,
        "mean": 25.328063062631585,
        "std": 245.96139201611473
    },
    "flow_latency_hvnet_p65": {
        "min": 29.763,
        "max": 3430.775,
        "mean": 210.49786669590645,
        "std": 357.158624714988
    },
    "flow_latency_omnet_p65": {
        "min": 3.386,
        "max": 3419.50485475,
        "mean": 25.590455271900588,
        "std": 246.40006330683772
    },
    "flow_latency_hvnet_p70": {
        "min": 30.062,
        "max": 3819.1,
        "mean": 220.97378751461991,
        "std": 370.89093343766274
    },
    "flow_latency_omnet_p70": {
        "min": 3.386,
        "max": 3425.2832049999997,
        "mean": 25.90210294040936,
        "std": 246.80373512950592
    },
    "flow_latency_hvnet_p75": {
        "min": 30.412,
        "max": 3860.449,
        "mean": 234.92806242690057,
        "std": 390.04155886407483
    },
    "flow_latency_omnet_p75": {
        "min": 3.386,
        "max": 3430.7042545000004,
        "mean": 26.272016634429832,
        "std": 247.17814471290873
    },
    "flow_latency_hvnet_p80": {
        "min": 30.875,
        "max": 3904.437,
        "mean": 250.12219608187135,
        "std": 417.82131867538385
    },
    "flow_latency_omnet_p80": {
        "min": 3.386,
        "max": 3435.7526860000003,
        "mean": 26.71509779128655,
        "std": 247.52242817629568
    },
    "flow_latency_hvnet_p85": {
        "min": 31.7,
        "max": 4229.0639,
        "mean": 263.9491918274854,
        "std": 435.6649570487105
    },
    "flow_latency_omnet_p85": {
        "min": 3.386,
        "max": 3440.48859765,
        "mean": 27.27144498688597,
        "std": 247.83671538690865
    },
    "flow_latency_hvnet_p90": {
        "min": 33.562,
        "max": 4355.1530999999995,
        "mean": 280.58806719298246,
        "std": 450.8986550724089
    },
    "flow_latency_omnet_p90": {
        "min": 3.386,
        "max": 3444.9939336,
        "mean": 28.028537462953228,
        "std": 248.1252012170677
    },
    "flow_latency_hvnet_p95": {
        "min": 40.125,
        "max": 4485.9166,
        "mean": 304.93713410818714,
        "std": 471.6984569587886
    },
    "flow_latency_omnet_p95": {
        "min": 3.386,
        "max": 3449.45503225,
        "mean": 29.29764718950292,
        "std": 248.37605080943598
    },
    "flow_latency_hvnet_p99": {
        "min": 58.79116,
        "max": 543682.4380000001,
        "mean": 850.3847449619892,
        "std": 15944.754280587655
    },
    "flow_latency_omnet_p99": {
        "min": 3.386,
        "max": 3457.1238335000003,
        "mean": 32.17330318411404,
        "std": 248.6431779548019
    },
    "flow_latency_hvnet_p99.9": {
        "min": 74.01240000000071,
        "max": 1260998.8344000033,
        "mean": 1960.383269577812,
        "std": 42065.0708329345
    },
    "flow_latency_omnet_p99.9": {
        "min": 3.386,
        "max": 3466.509921886001,
        "mean": 36.3534001319154,
        "std": 249.00368335076865
    },
    "flow_latency_hvnet_p99.99": {
        "min": 93.96812819999994,
        "max": 1337844.8655599903,
        "mean": 4276.929442647965,
        "std": 50985.42148227718
    },
    "flow_latency_omnet_p99.99": {
        "min": 5.296854852599892,
        "max": 3475.3149257689975,
        "mean": 40.27146966857461,
        "std": 249.40155994732868
    },
    "flow_latency_hvnet_p99.999": {
        "min": 96.91391431999321,
        "max": 1643777.9822396105,
        "mean": 10436.833911201162,
        "std": 96416.20166011255
    },
    "flow_latency_omnet_p99.999": {
        "min": 5.7170667323799425,
        "max": 3483.6790534090496,
        "mean": 43.41103602438036,
        "std": 249.77900530735812
    },
    "flow_latency_hvnet_p99.9999": {
        "min": 97.90309143199498,
        "max": 1644519.4809218731,
        "mean": 11585.366077169203,
        "std": 105206.98135574425
    },
    "flow_latency_omnet_p99.9999": {
        "min": 5.725429773237909,
        "max": 3495.018997717543,
        "mean": 45.25038909516328,
        "std": 250.12611042245325
    }
}

OM2HVNetNodeType = enum.IntEnum("NodeType", [
    "Link",
    "SelfLink",  # termination links, as those don't have link rates
    "Flow",
    "Path",  # i.e., direction
])


def log_normalize(x, param_name=None, mapping=None):
    return np.log(1 + x)


def log_denormalize(y, param_name=None, mapping=None):
    return np.exp(y) - 1


def min_max_normalize(x, param_name, min_max_mapping=PARAMS_NORMALIZATION_VALUES_OM2HVNET):
    min_param = min_max_mapping[param_name]['min']
    max_param = min_max_mapping[param_name]['max']
    return (x - min_param) / (max_param - min_param)


def min_max_denormalize(y, param_name, min_max_mapping=PARAMS_NORMALIZATION_VALUES_OM2HVNET):
    min_param = min_max_mapping[param_name]['min']
    max_param = min_max_mapping[param_name]['max']
    return y * (max_param - min_param) + min_param


def z_normalize(x, param_name, z_mapping=PARAMS_NORMALIZATION_VALUES_OM2HVNET):
    mean_param = z_mapping[param_name]['mean']
    std_param = z_mapping[param_name]['std']
    if math.isnan(x):
        return float('nan')
    if x == mean_param and std_param == 0.0:
        return 1
    return (x - mean_param) / std_param


def z_denormalize(y, param_name, z_mapping=PARAMS_NORMALIZATION_VALUES_OM2HVNET):
    mean_param = z_mapping[param_name]['mean']
    std_param = z_mapping[param_name]['std']
    return y * std_param + mean_param


def get_om2hvnet_features(target='quantiles'):
    feature_list = []
    if target == 'quantiles':
        feature_list += ['mean']
        feature_list += ['min']
        feature_list += ['p25', 'p50', 'p75', 'p95', 'p99', 'p99.9', 'p99.99', 'p99.999']
    elif target == 'gamma_params':
        feature_list += ['gamma_shape', 'gamma_scale']
    elif target == 'mean':
        feature_list += ['mean']
    else:
        raise ValueError(f'Unknown target {target}')
    return feature_list


def get_om2hvnet_normalization_function(normalization='log-normalize'):
    if normalization == 'log-normalize':
        return log_normalize
    elif normalization == 'min_max_normalize':
        return min_max_normalize
    elif normalization == 'z_normalize':
        return z_normalize
    else:
        raise ValueError(f'Unknown normalization {normalization}')


def get_om2hvnet_denormalization_function(normalization='log-normalize'):
    if normalization == 'log-normalize':
        return log_denormalize
    elif normalization == 'min_max_normalize':
        return min_max_denormalize
    elif normalization == 'z_normalize':
        return z_denormalize
    else:
        raise ValueError(f'Unknown normalization {normalization}')


DEFAULT_WEIGHTS = [
    2.,  # min
    1.5,  # <= p25
    1.,  # <= p75
    .5,  # <= p95
    .1,  # <= p99
    .01,  # <= p99.9
    .001,  # <= p99.99
    .0001,  # <= p99.999
    .00001  # > p99.999
]


def get_om2hvnet_loss_weights(args, target='quantiles', custom_weights=None):
    if target != 'quantiles':
        raise ValueError(f'Target {target} does not support loss weights')
    weights = []
    for f in get_om2hvnet_features(target):
        if f.startswith('p'):
            quantile = float(f[1:])
            if quantile <= 25:
                weights.append(args.loss_weight_le_p25)
            elif quantile <= 75:
                weights.append(args.loss_weight_le_p75)
            elif quantile <= 95:
                weights.append(args.loss_weight_le_p95)
            elif quantile <= 99:
                weights.append(args.loss_weight_le_p99)
            elif quantile <= 99.9:
                weights.append(args.loss_weight_le_p99_9)
            elif quantile <= 99.99:
                weights.append(args.loss_weight_le_p99_99)
            elif quantile <= 99.999:
                weights.append(args.loss_weight_le_p99_999)
            else:
                weights.append(args.loss_weight_gt_p99_999)
        elif f == 'min':
            weights.append(args.loss_weight_min)
    return torch.Tensor(weights).to(device=torch.device(args.device))
