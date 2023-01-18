
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor

config = Config('/home/ethan/Downloads/PaddleOCRv3/en_ppocr_mobile_v2.0_table_structure_infer/inference.pdmodel', '/home/ethan/Downloads/PaddleOCRv3/en_ppocr_mobile_v2.0_table_structure_infer/inference.pdiparams')
config.enable_memory_optim()
config.set_cpu_math_library_num_threads(4)
config.enable_mkldnn()

predictor = create_predictor(config)

def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results
data = np.ones((1,3,488,488))
data  = data.astype('float32')
#data = np.ones((1,3,488,488), dtype=float)
result = run(predictor, [data])
result