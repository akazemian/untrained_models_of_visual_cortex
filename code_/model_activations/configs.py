model_cfg = {
    "things": {
        "models": {
            "vit": {"features": [12, 12 * 5, 12 * 50], "layers": None},
            "expansion": {"features": [3, 30, 300, 3000], "layers": 5},
            "expansion_linear": {"features": [3, 30, 300, 3000], "layers": 5},
            "fully_connected": {"features": [108, 1080, 10800, 108000], "layers": 5},
            "fully_random": {"features": [3, 30, 300, 3000], "layers": 5},
        },
        "regions": ["V1", "V2", "V3", "V4", "FFA", "PPA", "LOC"],
        "subjects": [i for i in range(3)],
        "test_data_size": 1482
    },    
    "naturalscenes": {
        "models": {
            "vit": {"features": [12, 12 * 5, 12 * 50], "layers": None},
            "expansion": {"features": [3, 30, 300, 3000], "layers": 5},
            "expansion_linear": {"features": [3, 30, 300, 3000], "layers": 5},
            "fully_connected": {"features": [108, 1080, 10800, 108000], "layers": 5},
            "fully_random": {"features": [3, 30, 300, 3000], "layers": 5},
        },
        "regions": ["early visual stream",
         "midventral visual stream", "ventral visual stream"],
        "subjects": [i for i in range(8)],
        "test_data_size": 872
    },
    "majajhong": {
        "models": {
            "vit": {"features": [12, 12 * 5, 12 * 50, 12 * 500], "layers": None},
            "expansion": {"features": [3, 30, 300, 3000, 30000], "layers": 5},
            "expansion_linear": {"features": [3, 30, 300, 3000, 30000], "layers": 5},
            "fully_connected": {"features": [108, 1080, 10800, 108000, 1080000], "layers": 5},
            "fully_random": {"features": [3, 30, 300, 3000, 30000], "layers": 5},
        },
        "regions": ["V4", "IT"],
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 640
    },
    "majajhong_demo": {
        "models": {
            "vit": {"features": [12, 12 * 5, 12 * 50], "layers": None},
            "expansion": {"features": [3, 30, 300], "layers": 5},
            "fully_connected": {"features": [108, 1080, 10800], "layers": 5},
            "fully_random": {"features": [3, 30, 300], "layers": 5},
             "expansion_linear": {"features": [3, 30, 300], "layers": 5},
        },
        "regions": ["V4","IT"],
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    },

}

analysis_cfg = {
    "things": {
        "analysis": {
            "pca": {"features": [3, 30, 300, 3000], "layers": 5},
            "non_linearities": {"features": [3000], "layers": 5, "variations": ["gelu", "elu", "abs", "leaky_relu"]},
            "init_types": {"features": [3, 30, 300, 3000], "layers": 5, 
                           "variations": ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"]}
        },
        "regions": "ventral visual stream",
        "subjects": [i for i in range(3)],
        "test_data_size": 1482
    },
     "naturalscenes": {
        "analysis": {
            "pca": {"features": [3, 30, 300, 3000], "layers": 5},
            "non_linearities": {"features": [3000], "layers": 5, "variations": ["gelu", "elu", "abs", "leaky_relu"]},
            "init_types": {"features": [3, 30, 300, 3000], "layers": 5, 
                           "variations": ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"]}
        },
        "regions": "ventral visual stream",
        "subjects": [i for i in range(8)],
        "test_data_size": 872
    },
    "naturalscenes_shuffled": {
        "models": {
            "expansion": {"features": [3, 30, 300, 3000], "layers": 5}
        },
        "regions": "ventral visual stream",
        "subjects": [i for i in range(8)],
        "test_data_size": 872
    },
    "majajhong": {
        "analysis": {
            "pca": {"features": [3, 30, 300, 3000, 30000], "layers": 5},
            "non_linearities": {"features": [30000], "layers": 5, "variations": ["gelu", "elu", "abs", "leaky_relu"]},
            "init_types": {"features": [3, 30, 300, 3000, 30000], "layers": 5, 
                           "variations": ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"]}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 640
    },
    "majajhong_shuffled": {
        "models": {
            "expansion": {"features": [3, 30, 300, 3000, 30000], "layers": 5}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 640
    },


    "places_val": {
        "models": {
            "alexnet_trained": {"features": None, "layers": 5},
            "expansion": {"features": 3000, "layers": 5},
        },
    },
    "places_train": {
        "models": {
            "alexnet_trained": {"features": None, "layers": 5},
            "expansion": {"features": 3000, "layers": 5},
        },
    },

    "majajhong_demo": {
        "analysis": {
            "activation_function": {"features": [3], "layers": 5},
            "pca": {"features": [3, 30, 300], "layers": 5},
            "non_linearities": {"features": [3], "layers": 5, "variations": ["relu", "gelu", "elu", "abs", "leaky_relu"]},
            "init_types": {"features": [3, 30, 300], "layers": 5, 
                           "variations": ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"]}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    },
    "majajhong_demo_shuffled": {
        "models": {
            "expansion": {"features": [3, 30, 300], "layers": 5}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    },
    "places_val_demo": {
        "models": {
            "alexnet_trained": {"features": None, "layers": 5},
            "expansion": {"features": 3, "layers": 5},
        },
    },
    "places_train_demo": {
        "models": {
            "alexnet_trained": {"features": None, "layers": 5},
            "expansion": {"features": 3, "layers": 5},
        },
    }
}