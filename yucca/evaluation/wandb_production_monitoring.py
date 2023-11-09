# %%
# TO BE IMPLEMENTED

# from weave.monitoring import StreamTable
# from batchgenerators.utilities.file_and_folder_operations import load_json, join
# import wandb
# from yucca.paths import yucca_results
#
# def add_row_to_streamtable(table_name: str):
#    st = StreamTable(table_name="table_name",
#                    entity_name=wandb.api.viewer()['entity'],
#                    project_name='Yucca')
#
# metrics_to_keep = ["Dice", "Sensitivity", "Precision", "Volume Similarity"]
#
# test_d_selected = test_d['mean']
# new_d = {"0. Model": "NAMES2",
#         "0. Source": "Results.json"}
# for key, val in test_d_selected.items():
#    if key == "0":
#        continue
#    else:
#        print(test_d_selected)
#        new_d.update({f"{key}. "+k:v for k, v in test_d_selected[key].items() if k in metrics_to_keep})
#
# st.log(new_d)
# st.finish()
