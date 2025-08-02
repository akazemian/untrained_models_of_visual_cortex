import os
import runpy

PATH = f'/home/akazemi3/Desktop/untrained_models_of_visual_cortex/model_evaluation/predicting_brain_data'



for score_type in ['shuffled_pixels','custom_models','non_linearity','init_type','alexnet']: # get brain score for all analysis

    script_path = os.path.join(PATH,f'{score_type}_score.py')
    runpy.run_path(script_path)
