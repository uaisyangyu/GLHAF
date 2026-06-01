from run import G2L_run

G2L_run(model_name='G2L', dataset_name='mosei', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_training=False)




