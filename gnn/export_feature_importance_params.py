import os
import enum
import yaml

from utils import *

EXPORT_FOLDER = './feature-importance-params'
os.makedirs(EXPORT_FOLDER, exist_ok=True)

OM2HVNET_PREDICTION_TARGET = 'quantiles'
OM2HVNET_FEATURES = get_om2hvnet_features(OM2HVNET_PREDICTION_TARGET)


node_parameters = {
    'ntype': {
        'encoding': len(OM2HVNetNodeType) + 1, 'is_y': False
    }, 'flow_rate': {
        'encoding': 0, 'is_y': False
    }, 'flow_gamma_shape': {
        'encoding': 0, 'is_y': False
    }, 'flow_gamma_scale': {
        'encoding': 0, 'is_y': False
    }, 'hvnet_link_rate': {
        'encoding': 0, 'is_y': False
    }, 'omnet_link_rate': {
        'encoding': 0, 'is_y': False
    }
}

node_parameters_aggregated = {
    'ntype': {
        'encoding': len(OM2HVNetNodeType) + 1, 'is_y': False
    }, 'flow_rate': {
        'encoding': 0, 'is_y': False
    }, 'flow_gamma_shape_scale': {
        'encoding': 2, 'is_y': False
    }, 'hvnet_link_rate': {
        'encoding': 0, 'is_y': False
    }, 'omnet_link_rate': {
        'encoding': 0, 'is_y': False
    }
}

# Add OMNeT++ features
for f in OM2HVNET_FEATURES:
    param_name = f'flow_latency_omnet_{f}'
    node_parameters[param_name] = {
        'encoding': 0, 'is_y': False
    }
if OM2HVNET_PREDICTION_TARGET == 'quantiles':
    node_parameters_aggregated['flow_latency_percentiles_omnet'] = {
        'encoding': len(OM2HVNET_FEATURES), 'is_y': False
    }

# Add HVNet features
for f in OM2HVNET_FEATURES:
    param_name = f'flow_latency_hvnet_{f}'
    node_parameters[param_name] = {
        'encoding': 0, 'is_y': True
    }
if OM2HVNET_PREDICTION_TARGET == 'quantiles':
    node_parameters_aggregated['flow_latency_percentiles_hvnet'] = {
        'encoding': len(OM2HVNET_FEATURES), 'is_y': True
    }

with open(os.path.join(EXPORT_FOLDER, 'sim2hw_feature_importance_params.yml'), 'w') as f:
    yaml.dump(node_parameters, f, sort_keys=False)
with open(os.path.join(EXPORT_FOLDER, 'sim2hw_feature_importance_params_aggregated.yml'), 'w') as f:
    yaml.dump(node_parameters_aggregated, f, sort_keys=False)
