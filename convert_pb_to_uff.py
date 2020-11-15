import uff

frozen_filename ='traffic_sign_classification_resnet18_64.pb'
output_node_names = ['dense/Softmax']
output_uff_filename = 'traffic_sign_classification_resnet18_64.uff'

uff_mode = uff.from_tensorflow_frozen_model(frozen_filename, output_nodes=output_node_names, output_filename=output_uff_filename, text=False)
