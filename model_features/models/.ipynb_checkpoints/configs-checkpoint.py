model_cfg = { 
    "naturalscenes":{
        "models":{
            "ViT":{"features":[12, 12*5, 12*50], "layers":None},
            "expansion":{"features":[3,30,300,3000], "layers":5},
            "expansion_linear":{"features":[3,30,300,3000], "layers":5},
            "fully_connected":{"features":[3,30,300,3000], "layers":5},
            "fully_random":{"features":[3,30,300,3000], "layers":5},
        },
        "regions":["midventral visual stream", "ventral visual stream","early visual stream"],
        "subjects":[i for i in range(8)],
        "test_data_size":872
    },
    "majajhong":{       
        "models":{
            "ViT":{"features":[12, 12*5, 12*50, 12*500], "layers":None},
            "expansion":{"features":[3,30,300,3000,30000], "layers":5},
            "expansion":{"features":[3,30,300,3000,30000], "layers":5},
            "expansion_linear":{"features":[3,30,300,3000,30000], "layers":5},
            "fully_connected":{"features":[3,30,300,3000,30000], "layers":5},
            "fully_random":{"features":[3,30,300,3000,30000], "layers":5},
        },
    "regions":["V4","IT"],
    "subjects":["Tito","Chabo"],
    "test_data_size":640
    }
}




analysis_cfg = { 
    "naturalscenes":{
        "analysis":{
            "pca":{"features":[3,30,300,3000], "layers":5},
            "non_linearities":{"features":[3000], "layers":5, "variations":["gelu","elu","abs","leaky_relu"]},
            "init_types":{"features":[3,30,300,3000], "layers":5, "variations":["kaiming_uniform","kaiming_normal","xavier_uniform","xavier_normal","orthogonal"]
    },
        },
        "regions":"ventral visual stream",
        "subjects":[i for i in range(8)],
        "test_data_size":872
    },
    "naturalscenes_shuffled":{
        "models":{
            "expansion":{"features":[3,30,300,3000], "layers":5}
        },
        "regions":"ventral visual stream",
        "subjects":[i for i in range(8)],
        "test_data_size":872
    },

    "majajhong":{       
        "analysis":{
            "pca":{"features":[3,30,300,3000, 30000], "layers":5},
            "non_linearities":{"features":[30000], "layers":5, "variations":["gelu","elu","abs","leaky_relu"]},
            "init_types":{"features":[3,30,300,3000, 30000], "layers":5, "variations":["kaiming_uniform","kaiming_normal","xavier_uniform","xavier_normal","orthogonal"]},
        },
        "regions":"IT",
        "subjects":["Tito","Chabo"],
        "test_data_size":640
    },
    "majajhong_shuffled":{     
        "models":{
            "expansion":{"features":[3,30,300,3000,30000], "layers":5}
        },
        "regions":"IT",
        "subjects":["Tito","Chabo"],
        "test_data_size":640
    }
}